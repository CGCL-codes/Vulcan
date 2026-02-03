import torch
import os
from tqdm import tqdm
import json
import math

def erank(W):
    S = torch.linalg.svdvals(W)
    p = S/(S.sum() + 1e-12)
    H = - (p * torch.log(p + 1e-12)).sum()
    erank = torch.exp(H)
    return erank.item()

def mha_adaptive_arch_config(args,model):
    root = os.path.join(args.output_dir,"config",args.model_name)
    file_name = f"qk_vo_dim_list({args.pruning_rate:.1f}).json"
    os.makedirs(root,exist_ok=True)
    config_path = os.path.join(root,file_name)
    if os.path.exists(config_path):
        with open(config_path,"r") as f:
            qk_dim_list,vo_dim_list = json.load(f)
        return qk_dim_list,vo_dim_list

    if hasattr(model, "blocks"):
        depth = len(model.blocks)
        num_heads = model.blocks[0].attn.num_heads
        head_dim = model.blocks[0].attn.head_dim
        query_size = model.blocks[0].attn.qkv.out_features//3
        hidden_size = model.blocks[0].attn.qkv.in_features
        WQKV_list = [model.blocks[l].attn.qkv.weight for l in range(depth)]
        bQKV_list = [model.blocks[l].attn.qkv.bias for l in range(depth)]
        WO_list = [model.blocks[l].attn.proj.weight for l in range(depth)]
        # load erank_list
        erank_name = f"erank_list.json"
        erank_path = os.path.join(root,erank_name)
        if os.path.exists(erank_path):
            with open(erank_path,"r") as f:
                qk_erank_list,vo_erank_list,erank_list = json.load(f)
        else:
            qk_erank_list = []
            vo_erank_list = []
            erank_list = []
            print("Calculate erank...")
            for l in tqdm(range(depth)):
                WQ = WQKV_list[l][:query_size].reshape(num_heads,head_dim,hidden_size)
                bQ = bQKV_list[l][:query_size].reshape(num_heads,head_dim)
                WK = WQKV_list[l][query_size:query_size*2].reshape(num_heads,head_dim,hidden_size)
                bK = bQKV_list[l][query_size:query_size*2].reshape(num_heads,head_dim)
                WV = WQKV_list[l][query_size*2:].reshape(num_heads,head_dim,hidden_size)
                bV = bQKV_list[l][query_size*2:].reshape(num_heads,head_dim)
                WO = WO_list[l].T.reshape(num_heads,head_dim,hidden_size)
                qk_erank_val = 0.0
                vo_erank_val = 0.0
                for h in range(num_heads):
                    WQ_hat = torch.cat((WQ[h], bQ[h].unsqueeze(1)), dim=1)
                    WK_hat = torch.cat((WK[h], bK[h].unsqueeze(1)), dim=1)
                    WV_hat = torch.cat((WV[h], bV[h].unsqueeze(1)), dim=1)
                    qk_erank_val += erank(WQ_hat.T @ WK_hat)
                    vo_erank_val += erank(WV_hat.T @ WO[h])
                qk_erank_list.append(qk_erank_val)
                vo_erank_list.append(vo_erank_val)
                erank_list.append(qk_erank_val+vo_erank_val)
            with open(erank_path,"w") as f:
                json.dump([qk_erank_list,vo_erank_list,erank_list],f)

        # get the qk_dim_list, vo_dim_list
        pruning_rate = args.pruning_rate
        qk_dim_list = []
        vo_dim_list = []
        for l in range(depth):
            # rate = 1-(1-pruning_rate)*depth*erank_list[l]/sum(erank_list)
            rate = pruning_rate*depth/sum(1/x for x in erank_list)/erank_list[l]
            # qk erank and vo erank are different
            qk_erank = qk_erank_list[l]
            vo_erank = vo_erank_list[l]
            qk_rate = rate*(2*vo_erank)/(qk_erank+vo_erank)
            vo_rate = rate*(2*qk_erank)/(qk_erank+vo_erank)
            qk_dim = int(head_dim*(1-qk_rate))
            vo_dim = int(head_dim*(1-vo_rate))
            qk_dim = (qk_dim//8)*8 if qk_dim>8 else 8
            vo_dim = (vo_dim//8)*8 if vo_dim>8 else 8
            qk_dim_list.append(qk_dim)
            vo_dim_list.append(vo_dim)

        with open(config_path,"w") as f:
            json.dump([qk_dim_list,vo_dim_list],f)

    elif hasattr(model, "backbone"):
        # load erank_list
        erank_name = f"erank_list.json"
        erank_path = os.path.join(root,erank_name)
        if os.path.exists(erank_path):
            with open(erank_path,"r") as f:
                qk_erank_list,vo_erank_list,erank_list = json.load(f)
        else:
            stage_num = len(model.backbone.stages)
            qk_erank_list = [[] for _ in range(stage_num)]
            vo_erank_list = [[] for _ in range(stage_num)]
            erank_list = [[] for _ in range(stage_num)]
            print("Calculate erank...")
            for s,stage in enumerate(model.backbone.stages):
                for b,block in enumerate(stage.blocks):
                    num_heads = block.attn.w_msa.num_heads
                    embed_dim = block.attn.w_msa.embed_dims
                    WQKV = block.attn.w_msa.qkv.weight.data
                    bQKV = block.attn.w_msa.qkv.bias.data
                    WO = block.attn.w_msa.proj.weight.data
                    WQ = WQKV[:embed_dim].reshape(num_heads,-1,embed_dim)
                    bQ = bQKV[:embed_dim].reshape(num_heads,-1)
                    WK = WQKV[embed_dim:embed_dim*2].reshape(num_heads,-1,embed_dim)
                    bK = bQKV[embed_dim:embed_dim*2].reshape(num_heads,-1)
                    WV = WQKV[embed_dim*2:].reshape(num_heads,-1,embed_dim)
                    bV = bQKV[embed_dim*2:].reshape(num_heads,-1)
                    WO = WO.T.reshape(num_heads,-1,embed_dim)
                    qk_erank_val = 0.0
                    vo_erank_val = 0.0
                    for h in range(num_heads):
                        WQ_hat = torch.cat((WQ[h], bQ[h].unsqueeze(1)), dim=1)
                        WK_hat = torch.cat((WK[h], bK[h].unsqueeze(1)), dim=1)
                        WV_hat = torch.cat((WV[h], bV[h].unsqueeze(1)), dim=1)
                        qk_erank_val += erank(WQ_hat.T @ WK_hat)
                        vo_erank_val += erank(WV_hat.T @ WO[h])
                    qk_erank_list[s].append(qk_erank_val)
                    vo_erank_list[s].append(vo_erank_val)
                    erank_list[s].append(qk_erank_val+vo_erank_val)
            with open(erank_path,"w") as f:
                json.dump([qk_erank_list,vo_erank_list,erank_list],f)
            
        # get the qk_dim_list, vo_dim_list
        pruning_rate = args.pruning_rate
        qk_dim_list = [[] for _ in range(len(model.backbone.stages))]
        vo_dim_list = [[] for _ in range(len(model.backbone.stages))]
        W_times_S = 0.0
        total_param = 0
        for s,stage in enumerate(model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                w_msa = block.attn.w_msa
                num_param = w_msa.qkv.weight.data.numel()+w_msa.proj.weight.data.numel()
                total_param += num_param
                W_times_S += num_param*math.log(num_param)/erank_list[s][b]
        
        for s,stage in enumerate(model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                w_msa = block.attn.w_msa
                num_param = w_msa.qkv.weight.data.numel()+w_msa.proj.weight.data.numel()
                score = erank_list[s][b]/math.log(num_param)
                rate = pruning_rate*total_param/W_times_S/score
                # qk erank and vo erank are different
                head_dim = w_msa.embed_dims//w_msa.num_heads
                qk_erank = qk_erank_list[s][b]
                vo_erank = vo_erank_list[s][b]
                qk_rate = rate*(2*vo_erank)/(qk_erank+vo_erank)
                vo_rate = rate*(2*qk_erank)/(qk_erank+vo_erank)
                qk_dim = int(head_dim*(1-qk_rate))
                vo_dim = int(head_dim*(1-vo_rate))
                qk_dim = (qk_dim//8)*8 if qk_dim>8 else 8
                vo_dim = (vo_dim//8)*8 if vo_dim>8 else 8
                qk_dim_list[s].append(qk_dim)
                vo_dim_list[s].append(vo_dim)

        with open(config_path,"w") as f:
            json.dump([qk_dim_list,vo_dim_list],f)

    return qk_dim_list,vo_dim_list