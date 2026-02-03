import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional
import copy

# DeiT
from timm.layers.mlp import Mlp
from timm.layers.attention import Attention,maybe_add_mask
# Swin
from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.backbones.swin import WindowMSA

# pruning the MLP modules
@torch.no_grad
def pruning_ffn(model,clusters):
    '''
    clusters = [
        [
            {"anchor_neuron": 0, "neurons": [0, 1, 2, 3, ...]},
            {"anchor_neuron": 0, "neurons": [0, 1, 2, 3, ...]},
            ...
        ],
        [
            {"anchor_neuron": 0, "neurons": [0, 1, 2, 3, ...]},
            {"anchor_neuron": 0, "neurons": [0, 1, 2, 3, ...]},
            ...
        ],
        ...
    ]
    '''
    if hasattr(model, "blocks"):
        depth = len(model.blocks)
        W1_list = [model.blocks[l].mlp.fc1.weight for l in range(depth)]
        b1_list = [model.blocks[l].mlp.fc1.bias for l in range(depth)]
        W2_list = [model.blocks[l].mlp.fc2.weight for l in range(depth)]
        b2_list = [model.blocks[l].mlp.fc2.bias for l in range(depth)]
        intermediate_size = model.blocks[0].mlp.fc1.out_features
        intermediate_size_list = [
            len(clusters[l]) if clusters[l] is not None else intermediate_size 
            for l in range(depth)
        ]
        for l in range(depth):
            if clusters[l] is None:
                continue
            model.blocks[l].mlp = Mlp(
                in_features=model.blocks[l].mlp.fc1.in_features,
                hidden_features=intermediate_size_list[l],
                out_features=model.blocks[l].mlp.fc2.out_features,
            )
            for i in range(intermediate_size_list[l]):
                anchor_idx = clusters[l][i]["anchor_neuron"]
                neuron_idxs = clusters[l][i]["neurons"]
                model.blocks[l].mlp.fc1.weight[i] = W1_list[l][anchor_idx]
                model.blocks[l].mlp.fc1.bias[i] = b1_list[l][anchor_idx]
                model.blocks[l].mlp.fc2.weight[:,i] = W2_list[l][:,neuron_idxs].sum(dim=1)
            model.blocks[l].mlp.fc2.bias = b2_list[l]

    elif hasattr(model, "backbone"):
        for s,stage in enumerate(model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                if clusters[s][b] is None:
                    continue
                embed_dims = block.ffn.layers[0][0].in_features
                intermediate_size = len(clusters[s][b])
                W1 = copy.deepcopy(block.ffn.layers[0][0].weight.data)
                b1 = copy.deepcopy(block.ffn.layers[0][0].bias.data)
                W2 = copy.deepcopy(block.ffn.layers[1].weight.data)
                b2 = copy.deepcopy(block.ffn.layers[1].bias.data)
                block.ffn = FFN(
                    embed_dims=embed_dims,
                    feedforward_channels=intermediate_size,
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                )
                for i,c in enumerate(clusters[s][b]):
                    anchor_idx = c["anchor_neuron"]
                    neuron_idxs = c["neurons"]
                    block.ffn.layers[0][0].weight.data[i].copy_(W1[anchor_idx])
                    block.ffn.layers[0][0].bias.data[i].copy_(b1[anchor_idx])
                    block.ffn.layers[1].weight.data[:,i].copy_(W2[:,neuron_idxs].sum(dim=1))
                block.ffn.layers[1].bias.data.copy_(b2)

# pruning the MHA modules
class AttentionPruned(Attention):
    def __init__(self, hidden_size, num_heads, qk_dim, v_dim, qkv_bias=True):
        super().__init__((hidden_size//num_heads)*num_heads, num_heads, qkv_bias)
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.scale = self.qk_dim ** -0.5
        self.qkv = nn.Linear(hidden_size,  num_heads*(qk_dim*2+v_dim), bias=qkv_bias)
        self.proj = nn.Linear(v_dim*num_heads, hidden_size)

    def forward(self,x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x) # [B,N,num_heads*(qk_dim*2+v_dim)]
        q = qkv[:,:,:self.qk_dim*self.num_heads].reshape(B,N,self.num_heads,self.qk_dim).permute(0,2,1,3)
        k = qkv[:,:,self.qk_dim*self.num_heads:self.qk_dim*2*self.num_heads].reshape(B,N,self.num_heads,self.qk_dim).permute(0,2,1,3)
        v = qkv[:,:,self.qk_dim*2*self.num_heads:].reshape(B,N,self.num_heads,self.v_dim).permute(0,2,1,3)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.v_dim*self.num_heads)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowMSAPruned(WindowMSA):
    def __init__(self, embed_dims, num_heads, window_size, qk_dim, v_dim, qkv_bias=True):
        super().__init__(
            embed_dims=embed_dims, 
            num_heads=num_heads, 
            window_size=window_size, 
            qkv_bias=qkv_bias
        )
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.scale = self.qk_dim ** -0.5
        self.qkv = nn.Linear(embed_dims,  num_heads*(qk_dim*2+v_dim), bias=qkv_bias)
        self.proj = nn.Linear(v_dim*num_heads, embed_dims)

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        qkv = self.qkv(x) # [B,N,num_heads*(qk_dim*2+v_dim)]
        q = qkv[:,:,:self.qk_dim*self.num_heads].reshape(B,N,self.num_heads,self.qk_dim).permute(0,2,1,3)
        k = qkv[:,:,self.qk_dim*self.num_heads:self.qk_dim*2*self.num_heads].reshape(B,N,self.num_heads,self.qk_dim).permute(0,2,1,3)
        v = qkv[:,:,self.qk_dim*2*self.num_heads:].reshape(B,N,self.num_heads,self.v_dim).permute(0,2,1,3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.v_dim*self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def svd(W):
    try:
        U, Sigma, Vh = torch.linalg.svd(W)
    except RuntimeError:
        print("SVD failed, fallback to gesvd with regularization")
        eps = 1e-6 * torch.eye(W.shape[0], device=W.device)
        U, Sigma, Vh = torch.linalg.svd(W + eps, driver="gesvd")
    return U,Sigma,Vh

@torch.no_grad
def pruning_attn(model,qk_dim_list,vo_dim_list):
    if hasattr(model, "blocks"):
        depth = len(model.blocks)
        num_heads = model.blocks[0].attn.num_heads
        head_dim = model.blocks[0].attn.head_dim
        query_size = model.blocks[0].attn.qkv.out_features//3
        hidden_size = model.blocks[0].attn.qkv.in_features
        WQKV_list = [model.blocks[l].attn.qkv.weight.data for l in range(depth)]
        bQKV_list = [model.blocks[l].attn.qkv.bias.data for l in range(depth)]
        WO_list = [model.blocks[l].attn.proj.weight.data for l in range(depth)]
        bO_list = [model.blocks[l].attn.proj.bias.data for l in range(depth)]
        for l in range(depth):
            qk_dim = qk_dim_list[l]
            vo_dim = vo_dim_list[l]
            model.blocks[l].attn = AttentionPruned(
                hidden_size,
                num_heads=num_heads,
                qk_dim=qk_dim,
                v_dim=vo_dim,
                qkv_bias=True,
            )
            WQ = WQKV_list[l][:query_size].reshape(num_heads,head_dim,hidden_size)
            bQ = bQKV_list[l][:query_size].reshape(num_heads,head_dim)
            WK = WQKV_list[l][query_size:query_size*2].reshape(num_heads,head_dim,hidden_size)
            bK = bQKV_list[l][query_size:query_size*2].reshape(num_heads,head_dim)
            WV = WQKV_list[l][query_size*2:].reshape(num_heads,head_dim,hidden_size)
            bV = bQKV_list[l][query_size*2:].reshape(num_heads,head_dim)
            WO = WO_list[l].T.reshape(num_heads,head_dim,hidden_size)
            WQ_new_list,bQ_new_list = [],[]
            WK_new_list,bK_new_list = [],[]
            WV_new_list,bV_new_list = [],[]
            WO_new_list = []
            for h in range(num_heads):
                WQ_hat = torch.cat((WQ[h],bQ[h].unsqueeze(1)),dim=1)
                WK_hat = torch.cat((WK[h],bK[h].unsqueeze(1)),dim=1)
                WV_hat = torch.cat((WV[h],bV[h].unsqueeze(1)),dim=1)
                # pruning qk_size
                U, Sigma, Vh = svd(WQ_hat.T @ WK_hat)
                WQ_hat_new = (U[:,:qk_dim] @ torch.sqrt(torch.diag(Sigma[:qk_dim]))).T * math.sqrt(qk_dim/head_dim)
                WK_hat_new = torch.sqrt(torch.diag(Sigma[:qk_dim])) @ Vh[:qk_dim,:]
                WQ_new_list.append(WQ_hat_new[:,:-1])
                bQ_new_list.append(WQ_hat_new[:,-1].T)
                WK_new_list.append(WK_hat_new[:,:-1])
                bK_new_list.append(WK_hat_new[:,-1].T)
                # pruning vo_size
                U, Sigma, Vh = svd(WV_hat.T @ WO[h])
                WV_hat_new = (U[:,:vo_dim] @ torch.sqrt(torch.diag(Sigma[:vo_dim]))).T
                WO_hat_new = torch.sqrt(torch.diag(Sigma[:vo_dim])) @ Vh[:vo_dim,:]
                WV_new_list.append(WV_hat_new[:,:-1])
                bV_new_list.append(WV_hat_new[:,-1].T)
                WO_new_list.append(WO_hat_new)
            WQKV_new = torch.cat((torch.cat(WQ_new_list,dim=0),torch.cat(WK_new_list,dim=0),torch.cat(WV_new_list,dim=0)),dim=0)
            bQKV_new = torch.cat((torch.cat(bQ_new_list,dim=0),torch.cat(bK_new_list,dim=0),torch.cat(bV_new_list,dim=0)),dim=0)
            WO_new = torch.cat(WO_new_list,dim=0).T
            model.blocks[l].attn.qkv.weight.data.copy_(WQKV_new)
            model.blocks[l].attn.qkv.bias.data.copy_(bQKV_new)
            model.blocks[l].attn.proj.weight.data.copy_(WO_new)
            model.blocks[l].attn.proj.bias.data.copy_(bO_list[l])

    elif hasattr(model, "backbone"):
        for s,stage in enumerate(model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                qk_dim = qk_dim_list[s][b]
                vo_dim = vo_dim_list[s][b]
                embed_dims = block.attn.w_msa.embed_dims
                num_heads = block.attn.w_msa.num_heads
                window_size = block.attn.w_msa.window_size
                head_dim = embed_dims // num_heads
                WQKV = copy.deepcopy(block.attn.w_msa.qkv.weight.data)
                bQKV = copy.deepcopy(block.attn.w_msa.qkv.bias.data)
                WO = copy.deepcopy(block.attn.w_msa.proj.weight.data)
                bO = copy.deepcopy(block.attn.w_msa.proj.bias.data)
                relative_position_bias_table = copy.deepcopy(block.attn.w_msa.relative_position_bias_table.data)
                block.attn.w_msa = WindowMSAPruned(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    window_size=window_size,
                    qk_dim=qk_dim,
                    v_dim=vo_dim,
                    qkv_bias=True,
                )
                WQ = WQKV[:embed_dims].reshape(num_heads,head_dim,embed_dims)
                bQ = bQKV[:embed_dims].reshape(num_heads,head_dim)
                WK = WQKV[embed_dims:embed_dims*2].reshape(num_heads,head_dim,embed_dims)
                bK = bQKV[embed_dims:embed_dims*2].reshape(num_heads,head_dim)
                WV = WQKV[embed_dims*2:].reshape(num_heads,head_dim,embed_dims)
                bV = bQKV[embed_dims*2:].reshape(num_heads,head_dim)
                WO = WO.T.reshape(num_heads,head_dim,embed_dims)
                WQ_new_list,bQ_new_list = [],[]
                WK_new_list,bK_new_list = [],[]
                WV_new_list,bV_new_list = [],[]
                WO_new_list = []
                for h in range(num_heads):
                    WQ_hat = torch.cat((WQ[h],bQ[h].unsqueeze(1)),dim=1)
                    WK_hat = torch.cat((WK[h],bK[h].unsqueeze(1)),dim=1)
                    WV_hat = torch.cat((WV[h],bV[h].unsqueeze(1)),dim=1)
                    # pruning qk_size
                    U, Sigma, Vh = svd(WQ_hat.T @ WK_hat)
                    WQ_hat_new = (U[:,:qk_dim] @ torch.sqrt(torch.diag(Sigma[:qk_dim]))).T * math.sqrt(qk_dim/head_dim)
                    WK_hat_new = torch.sqrt(torch.diag(Sigma[:qk_dim])) @ Vh[:qk_dim,:]
                    WQ_new_list.append(WQ_hat_new[:,:-1])
                    bQ_new_list.append(WQ_hat_new[:,-1].T)
                    WK_new_list.append(WK_hat_new[:,:-1])
                    bK_new_list.append(WK_hat_new[:,-1].T)
                    # pruning vo_size
                    U, Sigma, Vh = svd(WV_hat.T @ WO[h])
                    WV_hat_new = (U[:,:vo_dim] @ torch.sqrt(torch.diag(Sigma[:vo_dim]))).T
                    WO_hat_new = torch.sqrt(torch.diag(Sigma[:vo_dim])) @ Vh[:vo_dim,:]
                    WV_new_list.append(WV_hat_new[:,:-1])
                    bV_new_list.append(WV_hat_new[:,-1].T)
                    WO_new_list.append(WO_hat_new)
                WQKV_new = torch.cat((torch.cat(WQ_new_list,dim=0),torch.cat(WK_new_list,dim=0),torch.cat(WV_new_list,dim=0)),dim=0)
                bQKV_new = torch.cat((torch.cat(bQ_new_list,dim=0),torch.cat(bK_new_list,dim=0),torch.cat(bV_new_list,dim=0)),dim=0)
                WO_new = torch.cat(WO_new_list,dim=0).T
                block.attn.w_msa.qkv.weight.data.copy_(WQKV_new)
                block.attn.w_msa.qkv.bias.data.copy_(bQKV_new)
                block.attn.w_msa.proj.weight.data.copy_(WO_new)
                block.attn.w_msa.proj.bias.data.copy_(bO)
                block.attn.w_msa.relative_position_bias_table.data.copy_(relative_position_bias_table)



def pruning_vit(
        model,clusters,qk_dim_list,vo_dim_list,
        no_ffn=False,no_mha=False
    ):
    print(f"Pruning...")
    if not no_ffn:
        pruning_ffn(model,clusters)
    if not no_mha:
        pruning_attn(model,qk_dim_list,vo_dim_list)
    return model
