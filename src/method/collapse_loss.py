import torch

def weight_collapse_loss(model,clusters,lambda1,lambda2,lora=False,anchor=True):
    '''
    lambda1,lambda2: Lagrange multiplier
    '''
    if hasattr(model, "blocks"):
        if not lora:
            depth = len(model.blocks)
            W1_list = [model.blocks[l].mlp.fc1.weight for l in range(depth)]
            b1_list = [model.blocks[l].mlp.fc1.bias for l in range(depth)]
        else:
            depth = len(model.base_model.model.blocks)
            W1_list,b1_list = [],[]
            for l in range(depth):
                fc1 = model.base_model.model.blocks[l].mlp.fc1
                W1 = fc1.base_layer.weight
                b1 = fc1.base_layer.bias
                delta_W1 = fc1.get_delta_weight("default")
                W1_list.append(W1 + delta_W1)
                b1_list.append(b1)
        loss = torch.zeros(1,device=W1_list[0].device)
        for l in range(depth):
            if clusters[l] is None: continue
            W1 = W1_list[l]
            b1 = b1_list[l]
            cnt = 0
            loss_cluster = torch.zeros(1,device=W1.device)
            for cluster in clusters[l]:
                if cluster["neurons"].numel() == 1: continue
                cnt += 1
                anchor_idx = cluster["anchor_neuron"]
                neuron_idxs = cluster["neurons"]
                if not anchor:
                    anchor_idx = neuron_idxs[torch.randint(0, len(neuron_idxs), (1,))]
                neuron_idxs = neuron_idxs[neuron_idxs != anchor_idx]
                # weight
                diff_w = W1[neuron_idxs] - W1[anchor_idx]
                loss_cluster += lambda1 * diff_w.abs().sum()
                loss_cluster += lambda2 * diff_w.pow(2).sum()
                # bias
                diff_b = b1[neuron_idxs] - b1[anchor_idx]
                loss_cluster += lambda1 * diff_b.abs().sum()
                loss_cluster += lambda2 * diff_b.pow(2).sum()
            loss += loss_cluster/(cnt+1e-10)

    elif hasattr(model, "backbone"):
        device = model.backbone.stages[0].blocks[0].ffn.layers[0][0].weight.device
        loss = torch.zeros(1,device=device)
        for s,stage in enumerate(model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                if clusters[s][b] is None: continue
                W1 = block.ffn.layers[0][0].weight
                b1 = block.ffn.layers[0][0].bias
                cnt = 0
                loss_cluster = torch.zeros(1,device=W1.device)
                for cluster in clusters[s][b]:
                    if cluster["neurons"].numel() == 1: continue
                    cnt += 1
                    anchor_idx = cluster["anchor_neuron"]
                    neuron_idxs = cluster["neurons"]
                    neuron_idxs = neuron_idxs[neuron_idxs != anchor_idx]
                    # weight
                    diff_w = W1[neuron_idxs] - W1[anchor_idx]
                    loss_cluster += lambda1 * diff_w.abs().sum()
                    loss_cluster += lambda2 * diff_w.pow(2).sum()
                    # bias
                    diff_b = b1[neuron_idxs] - b1[anchor_idx]
                    loss_cluster += lambda1 * diff_b.abs().sum()
                    loss_cluster += lambda2 * diff_b.pow(2).sum()
                loss += loss_cluster/(cnt+1e-10)
    
    return loss

def activation_collapse_loss(model,clusters,activation,lambda1,lambda2):
    if hasattr(model, "blocks"):
        depth = len(model.blocks)
    else:
        pass
    loss = torch.zeros(1,device=activation[0].device)
    for l in range(depth):
        if clusters[l] is None: continue
        cnt = 0
        loss_cluster = torch.zeros(1,device=activation[0].device)
        for cluster in clusters[l]:
            if cluster["neurons"].numel() == 1: continue
            cnt += 1
            anchor_idx = cluster["anchor_neuron"]
            neuron_idxs = cluster["neurons"]
            neuron_idxs = neuron_idxs[neuron_idxs != anchor_idx]
            diff_a = activation[l][neuron_idxs] - activation[l][anchor_idx]
            loss_cluster += lambda1 * diff_a.abs().sum()
            loss_cluster += lambda2 * diff_a.pow(2).sum()
        loss += loss_cluster/(cnt+1e-10)
    return loss

def truncated_nuclear_norm(W, r):
    try:
        S = torch.linalg.svdvals(W)
    except:
        return torch.tensor(0.0, device=W.device)
    if r < S.shape[-1]:
        return S[:,r:].sum()
    else:
        return torch.tensor(0.0, device=W.device)

def truncated_nuclear_norm_squared(W, r):
    try:
        S = torch.linalg.svdvals(W)
    except:
        return torch.tensor(0.0, device=W.device)
    if r < S.shape[-1]:
        return (S[:,r:].pow(2)).sum()
    else:
        return torch.tensor(0.0, device=W.device)


def rank_loss(model,qk_dim_list,vo_dim_list,lambda1,lambda2,lora=False):
    # calculate nuclear_norm of WQ
    if hasattr(model, "blocks"):
        if not lora:
            depth = len(model.blocks)
            num_heads = model.blocks[0].attn.num_heads
            head_dim = model.blocks[0].attn.head_dim
            query_size = model.blocks[0].attn.qkv.out_features//3
            hidden_size = model.blocks[0].attn.qkv.in_features
            WQKV_list = [model.blocks[l].attn.qkv.weight for l in range(depth)]
            bQKV_list = [model.blocks[l].attn.qkv.bias for l in range(depth)]
        else:
            depth = len(model.base_model.model.blocks)
            num_heads = model.base_model.model.blocks[0].attn.num_heads
            head_dim = model.base_model.model.blocks[0].attn.head_dim
            query_size = model.base_model.model.blocks[0].attn.qkv.out_features//3
            hidden_size = model.base_model.model.blocks[0].attn.qkv.in_features
            WQKV_list,bQKV_list = [],[]
            for l in range(depth):
                qkv = model.base_model.model.blocks[l].attn.qkv
                WQKV = qkv.base_layer.weight
                delta_WQKV = qkv.get_delta_weight("default")
                bQKV = qkv.base_layer.bias
                WQKV_list.append(WQKV+delta_WQKV)
                bQKV_list.append(bQKV)

        loss = torch.zeros(1,device=WQKV_list[0].device)
        for l in range(depth):
            qk_dim = qk_dim_list[l]
            vo_dim = vo_dim_list[l]
            WQ = WQKV_list[l][:query_size].reshape(num_heads,head_dim,hidden_size)
            bQ = bQKV_list[l][:query_size].reshape(num_heads,head_dim)
            WV = WQKV_list[l][query_size*2:].reshape(num_heads,head_dim,hidden_size)
            bV = bQKV_list[l][query_size*2:].reshape(num_heads,head_dim)

            WQ_hat = torch.cat((WQ, bQ.unsqueeze(2)), dim=2)   # [num_heads, head_dim, hidden_size+1]
            WV_hat = torch.cat((WV, bV.unsqueeze(2)), dim=2)   # [num_heads, head_dim, hidden_size+1]
            loss += lambda1*truncated_nuclear_norm(WQ_hat, qk_dim)
            loss += lambda2*truncated_nuclear_norm_squared(WQ_hat, qk_dim)
            loss += lambda1*truncated_nuclear_norm(WV_hat, vo_dim)
            loss += lambda2*truncated_nuclear_norm_squared(WV_hat, vo_dim)

        loss /= num_heads*depth

    elif hasattr(model, "backbone"):
        device = model.backbone.stages[0].blocks[0].ffn.layers[0][0].weight.device
        loss = torch.zeros(1,device=device)
        head_num = 0
        for s,stage in enumerate(model.backbone.stages):
            for b,block in enumerate(stage.blocks):
                qk_dim = qk_dim_list[s][b]
                vo_dim = vo_dim_list[s][b]
                num_heads = block.attn.w_msa.num_heads
                embed_dim = block.attn.w_msa.embed_dims
                head_dim = embed_dim//num_heads
                head_num += num_heads
                WQKV = block.attn.w_msa.qkv.weight
                bQKV = block.attn.w_msa.qkv.bias
                WQ = WQKV[:embed_dim].reshape(num_heads,head_dim,embed_dim)
                bQ = bQKV[:embed_dim].reshape(num_heads,head_dim)
                WV = WQKV[embed_dim*2:].reshape(num_heads,head_dim,embed_dim)
                bV = bQKV[embed_dim*2:].reshape(num_heads,head_dim)

                WQ_hat = torch.cat((WQ, bQ.unsqueeze(2)), dim=2)   # [num_heads, head_dim, hidden_size+1]
                WV_hat = torch.cat((WV, bV.unsqueeze(2)), dim=2)   # [num_heads, head_dim, hidden_size+1]
                loss += lambda1*truncated_nuclear_norm(WQ_hat, qk_dim)
                loss += lambda2*truncated_nuclear_norm_squared(WQ_hat, qk_dim)
                loss += lambda1*truncated_nuclear_norm(WV_hat, vo_dim)
                loss += lambda2*truncated_nuclear_norm_squared(WV_hat, vo_dim)
        loss/=head_num
    
    return loss

def weight_collapse_loss_fast(model, clusters, lambda1, lambda2, lora=False, anchor=True):
    depth = len(model.blocks)
    
    W1_list = [model.blocks[l].mlp.fc1.weight for l in range(depth)]
    b1_list = [model.blocks[l].mlp.fc1.bias for l in range(depth)]

    device = W1_list[0].device
    total_loss = torch.zeros(1, device=device)

    for l in range(depth):
        if clusters[l] is None:
            continue

        W1 = W1_list[l]
        b1 = b1_list[l]

        all_idxs = []
        all_anchor_idxs = []
        cnt = 0

        for cluster in clusters[l]:
            neuron_idxs = cluster["neurons"]
            if neuron_idxs.numel() <= 1:
                continue
            
            cnt += 1

            if anchor:
                anchor_idx = cluster["anchor_neuron"]
            else:
                anchor_idx = neuron_idxs[torch.randint(0, len(neuron_idxs), (1,))]

            diff_idxs = neuron_idxs[neuron_idxs != anchor_idx]

            all_idxs.append(diff_idxs)
            all_anchor_idxs.append(torch.full_like(diff_idxs, anchor_idx))

        if cnt == 0:
            continue

        all_idxs = torch.cat(all_idxs)
        all_anchor_idxs = torch.cat(all_anchor_idxs)

        diff_W = W1[all_idxs] - W1[all_anchor_idxs]
        diff_b = b1[all_idxs] - b1[all_anchor_idxs]

        loss_cluster = (
            lambda1 * (diff_W.abs().sum() + diff_b.abs().sum())
            + lambda2 * ((diff_W ** 2).sum() + (diff_b ** 2).sum())
        )
        total_loss += loss_cluster / (cnt + 1e-10)

    return total_loss

def truncated_tail_sigma_sum(W, r):
    """
    等价于原来的 truncated_nuclear_norm(W, r)
    W: [B, m, n]
    返回: 标量（已经在 batch 维度上求和）
    """
    B, m, n = W.shape
    WWT = W @ W.transpose(1, 2)          # [B, m, m]
    eigvals = torch.linalg.eigvalsh(WWT) # [B, m]
    eigvals = torch.clamp(eigvals, min=0.0)
    sigmas = torch.sqrt(eigvals)

    if r >= m:
        return torch.zeros(1, device=W.device)

    tail = sigmas[:, :m - r].sum()
    return tail


def truncated_tail_sigma_sq_sum(W, r):
    """
    等价于原来的 truncated_nuclear_norm_squared(W, r)
    W: [B, m, n]
    """
    B, m, n = W.shape
    WWT = W @ W.transpose(1, 2)          # [B, m, m]
    eigvals = torch.linalg.eigvalsh(WWT) # [B, m]

    eigvals = torch.clamp(eigvals, min=0.0)

    if r >= m:
        return torch.zeros(1, device=W.device)

    tail_sq = eigvals[:, :m - r].sum()
    return tail_sq


def rank_loss_fast(model, qk_dim_list, vo_dim_list, lambda1, lambda2, lora=False):
    depth = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads
    head_dim  = model.blocks[0].attn.head_dim
    query_size = model.blocks[0].attn.qkv.out_features // 3
    hidden_size = model.blocks[0].attn.qkv.in_features

    WQKV_list = [model.blocks[l].attn.qkv.weight for l in range(depth)]
    bQKV_list = [model.blocks[l].attn.qkv.bias  for l in range(depth)]

    device = WQKV_list[0].device
    total_loss = torch.zeros(1, device=device)

    for l in range(depth):
        qk_dim = qk_dim_list[l]
        vo_dim = vo_dim_list[l]

        WQ = WQKV_list[l][:query_size].reshape(num_heads, head_dim, hidden_size)
        bQ = bQKV_list[l][:query_size].reshape(num_heads, head_dim)
        WV = WQKV_list[l][query_size * 2:].reshape(num_heads, head_dim, hidden_size)
        bV = bQKV_list[l][query_size * 2:].reshape(num_heads, head_dim)

        WQ_hat = torch.cat((WQ, bQ.unsqueeze(2)), dim=2)  # [num_heads, head_dim, hidden+1]
        WV_hat = torch.cat((WV, bV.unsqueeze(2)), dim=2)

        loss_q = lambda1 * truncated_tail_sigma_sum(WQ_hat, qk_dim)
        loss_q += lambda2 * truncated_tail_sigma_sq_sum(WQ_hat, qk_dim)

        loss_v = lambda1 * truncated_tail_sigma_sum(WV_hat, vo_dim)
        loss_v += lambda2 * truncated_tail_sigma_sq_sum(WV_hat, vo_dim)

        total_loss += (loss_q + loss_v)

    total_loss /= (num_heads * depth)
    return total_loss