import torch
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
import math

def l2_norm(x):
    return x / x.norm(dim=-1, keepdim=True)

def min_max_norm(x):
    min = x.min()
    max = x.max()
    return (x-min)/(max-min)

def z_score_norm(x):
    mean = x.mean()
    std = x.std(unbiased=False)
    return (x-mean)/std

def softmax_norm(x):
    T = x.abs().max().clamp(min=1e-6)
    return torch.softmax(x/T,dim=-1)

def normalization(x,norm_type="l2"):
    if norm_type == "l2":
        return l2_norm(x)
    elif norm_type == "min_max":
        return min_max_norm(x)
    elif norm_type == "z_score":
        return z_score_norm(x)
    elif norm_type == "softmax":
        return softmax_norm(x)
    else:
        raise ValueError(f"Unknown norm type {norm_type}")

def get_activation(args,model,dataloader):
    # get config
    if hasattr(model, "blocks"):
        depth = len(model.blocks)
        intermediate_size = model.blocks[0].mlp.fc1.out_features
        # register hook
        batch_num = len(dataloader)
        activation = torch.zeros(depth,intermediate_size)
        def register_hook(layer_idx):
            def hook_fn(module, input, output):
                # output [b,N,e]
                activation[layer_idx] += (output.sum(dim=(0,1))/batch_num).detach().cpu()
            return hook_fn
        hooks = []
        for layer_idx in range(depth):
            hooks.append(model.blocks[layer_idx].mlp.act.register_forward_hook(register_hook(layer_idx)))
        # get activation
        print("Get activation ...")
        model.eval()
        model.to(args.device)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                xb = batch[0]
                xb = xb.to(args.device)
                model(xb)
        for hook in hooks:
            hook.remove()
        # save activation
        root = os.path.join(args.output_dir,"activation",args.dataset_name,args.model_name)
        file_name = f"activation({args.task_name}).pt"
        os.makedirs(root,exist_ok=True)
        path = os.path.join(root,file_name)
        torch.save(activation,path)

    elif hasattr(model, "backbone"):
        # register hook
        batch_num = len(dataloader)
        activation = [
            [torch.zeros(block.ffn.layers[1].in_features) for block in stage.blocks] 
            for stage in model.backbone.stages
        ]
        def register_hook(stage_idx,block_idx):
            def hook_fn(module, input, output):
                # output [b,N,e]
                activation[stage_idx][block_idx] += (output.sum(dim=(0,1))/batch_num).detach().cpu()
            return hook_fn
        hooks = []
        for stage_idx in range(len(model.backbone.stages)):
            for block_idx in range(len(model.backbone.stages[stage_idx].blocks)):
                hooks.append(
                    model.backbone.stages[stage_idx].blocks[block_idx].ffn.layers[0][1].register_forward_hook(
                        register_hook(stage_idx,block_idx)
                    )
                )
        # get activation
        print("Get activation ...")
        model.eval()
        model.to(args.device)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                model.test_step(batch)
        for hook in hooks:
            hook.remove()
        # save activation
        root = os.path.join(args.output_dir,"activation",args.dataset_name,args.model_name)
        file_name = f"activation({args.task_name}).pt"
        os.makedirs(root,exist_ok=True)
        path = os.path.join(root,file_name)
        torch.save(activation,path)

def mlp_adaptive_arch_config(args,model,dataloader,norm_type="l2"):
    if norm_type == "uniform":
        if hasattr(model, "blocks"):
            intermediate_size = model.blocks[0].mlp.fc1.out_features
            intermediate_size = int(intermediate_size*(1-args.pruning_rate))
            if intermediate_size>8:
                intermediate_size = (intermediate_size//8)*8
            else:
                intermediate_size = 8
            return [intermediate_size for _ in range(len(model.blocks))]
        elif hasattr(model, "backbone"):
            intermediate_size_list = []
            for stage in model.backbone.stages:
                intermediate_size = stage.blocks[0].ffn.layers[1].in_features
                intermediate_size = int(intermediate_size*(1-args.pruning_rate))
                if intermediate_size>8:
                    intermediate_size = (intermediate_size//8)*8
                else:
                    intermediate_size = 8
                intermediate_size_list.append([intermediate_size for _ in range(len(stage.blocks))])
            return intermediate_size_list
                
    # get_activation
    root = os.path.join(args.output_dir,"activation",args.dataset_name,args.model_name)
    file_name = f"activation({args.task_name}).pt"
    path = os.path.join(root, file_name)
    if not os.path.exists(path):
        get_activation(args,model,dataloader)
    activation = torch.load(path)
    if hasattr(model, "blocks"):
        # normalize activation
        # activation = activation / activation.norm(dim=-1,keepdim=True)
        for l in range(activation.size(0)):
            activation[l] = normalization(activation[l],norm_type)
        # get intermediate_size
        pruning_rate = args.pruning_rate
        act_flat = activation.view(-1)
        k = int(len(act_flat) * pruning_rate)
        threshold = torch.kthvalue(act_flat,k).values
        mask = activation >= threshold
        act_new = activation * mask
        intermediate_size_list = []
        for i in range(activation.size(0)):
            intermediate_size = (act_new[i]!=0).sum().item()
            # implement details
            if intermediate_size>8:
                intermediate_size = (intermediate_size//8)*8
            else:
                intermediate_size = 8
            intermediate_size_list.append(intermediate_size)
    elif hasattr(model, "backbone"):
        # normalize activation
        for stage_idx in range(len(model.backbone.stages)):
            for block_idx in range(len(model.backbone.stages[stage_idx].blocks)):
                intermediate_size = model.backbone.stages[stage_idx].blocks[block_idx].ffn.layers[0][0].weight.data.shape[0]
                activation[stage_idx][block_idx]/=math.log(intermediate_size)
                activation[stage_idx][block_idx] = normalization(activation[stage_idx][block_idx],norm_type)
        # get intermediate_size
        pruning_rate = args.pruning_rate
        all_tensors = [t for stage in activation for t in stage]   # 拉平成一维列表
        act_flat = torch.cat(all_tensors) 
        k = int(len(act_flat) * pruning_rate)
        threshold = torch.kthvalue(act_flat,k).values
        intermediate_size_list = []
        for s in range(len(model.backbone.stages)):
            intermediate_size_list.append([])
            for b in range(len(model.backbone.stages[s].blocks)):
                act = activation[s][b]
                mask = act >= threshold
                activation[s][b] = act*mask
                intermediate_size = (activation[s][b]!=0).sum().item()
                if intermediate_size>8:
                    intermediate_size = (intermediate_size//8)*8
                else:
                    intermediate_size = 8
                intermediate_size_list[s].append(intermediate_size)

    return intermediate_size_list

def get_clusters(args,model,dataloader,norm_type="l2"):
    root = os.path.join(args.output_dir,"cluster",args.dataset_name,args.model_name)
    file_name = f"clusters({args.task_name}-{args.pruning_rate:.1f}-{norm_type}).pt"
    os.makedirs(root,exist_ok=True)
    cluster_path = os.path.join(root,file_name)
    if os.path.exists(cluster_path):
        clusters = torch.load(cluster_path)
        return clusters
    print(f"Construct Clusters (norm_type={norm_type})...")
    # get config
    intermediate_size_list = mlp_adaptive_arch_config(args,model,dataloader,norm_type)
    # get_activation
    root = os.path.join(args.output_dir,"activation",args.dataset_name,args.model_name)
    file_name = f"activation({args.task_name}).pt"
    path = os.path.join(root, file_name)
    activation = torch.load(path)
    if hasattr(model, "blocks"):
        # get weight
        depth = len(model.blocks)
        intermediate_size = model.blocks[0].mlp.fc1.out_features
        weight_list = [model.blocks[l].mlp.fc1.weight.data for l in range(depth)]
        # get cluster
        clusters = [[] for _ in range(depth)]
        for l in tqdm(range(depth)):
            intermediate_size = intermediate_size_list[l]
            weight = weight_list[l]
            if weight.size(0) == intermediate_size:
                clusters[l] = None
                continue
            kmeans = KMeans(n_clusters=intermediate_size, random_state=0).fit(weight.numpy())
            labels = kmeans.labels_
            for c in range(intermediate_size):
                neuron_idxs = torch.tensor((labels == c).nonzero()[0])
                cluster_activations = activation[l][neuron_idxs]
                max_idx_in_cluster = torch.argmax(cluster_activations)
                anchor_idx = neuron_idxs[max_idx_in_cluster].item()
                clusters[l].append({
                    "anchor_neuron": anchor_idx,
                    "neurons": neuron_idxs,
                })
        torch.save(clusters,cluster_path)
    elif hasattr(model, "backbone"):
        stage_num = len(model.backbone.stages)
        clusters = [
            [[] for b in range(len(model.backbone.stages[s].blocks))] 
            for s in range(stage_num)
        ]
        for s in tqdm(range(stage_num)):
            for b in range(len(model.backbone.stages[s].blocks)):
                intermediate_size = intermediate_size_list[s][b]
                weight = model.backbone.stages[s].blocks[b].ffn.layers[0][0].weight.data
                if weight.size(0) == intermediate_size:
                    clusters[s][b] = None
                    continue
                kmeans = KMeans(n_clusters=intermediate_size, random_state=0).fit(weight.cpu().numpy())
                labels = kmeans.labels_
                for c in range(intermediate_size):
                    neuron_idxs = torch.tensor((labels == c).nonzero()[0])
                    cluster_activations = activation[s][b][neuron_idxs]
                    max_idx_in_cluster = torch.argmax(cluster_activations)
                    anchor_idx = neuron_idxs[max_idx_in_cluster].item()
                    clusters[s][b].append({
                        "anchor_neuron": anchor_idx,
                        "neurons": neuron_idxs,
                    })
        torch.save(clusters,cluster_path)
                
    return clusters
    