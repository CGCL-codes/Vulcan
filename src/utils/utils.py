import random
import numpy as np
import torch
import torch.nn as nn
import os
from thop import profile
from tqdm import tqdm
from time import perf_counter
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

'''load sub_task'''
def get_sub_task(args):
    path = os.path.join(args.output_dir,"sub_task",args.dataset_name,args.task_name+".txt")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    sub_task = [int(x.strip()) for x in content.split(",") if x.strip()]
    return sub_task

'''get lagrange multiplier'''
def get_lagrange_multiplier(model):
    model.lambda_1 = nn.Parameter(torch.zeros(1))
    model.lambda_2 = nn.Parameter(torch.zeros(1))

'''calculate GFLOPs'''
def get_metrics(model,input_size=(1,3,224,224)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        x = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(x,))
    print(f"GFLOPs: {flops/1e9:.2f}, MParams: {params/1e6:.2f}")
    return flops,params

'''get gflops'''
def is_identity_layer(attn: torch.nn.Module):
    # 没有任何参数
    has_params = any(p.numel() > 0 for p in attn.parameters())
    return not has_params
def get_gflops(model=None,config=None):
    if model is not None:
        N = model.pos_embed.shape[1]
        C = model.head.out_features
        p = model.patch_embed.patch_size[0]
        depth = len(model.blocks)
        for l in range(depth):
            if is_identity_layer(model.blocks[l]):
                continue
            emb_dim = model.blocks[l].attn.qkv.in_features
            break

        flops = (3*(N-1)*p*p+2*depth*N+C+1)*emb_dim
        for l in range(depth):
            # attn
            if is_identity_layer(model.blocks[l]):
                continue
            attn = model.blocks[l].attn
            if not is_identity_layer(attn):
                num_heads = attn.num_heads
                head_dim = attn.head_dim
                qk_dim = attn.qk_dim if hasattr(attn,"qk_dim") else head_dim
                v_dim = attn.v_dim if hasattr(attn,"v_dim") else head_dim
                flops += (2*N*emb_dim+N*N)*(qk_dim+v_dim)*num_heads
            # mlp
            mlp = model.blocks[l].mlp
            mlp_dim = mlp.fc1.out_features
            flops += 2*N*emb_dim*mlp_dim
    elif config is not None:
        N = config["N"]
        C = config["C"]
        p = config["p"]
        depth = config["depth"]
        emb_dim = config["emb"]
        flops = (3*(N-1)*p*p+2*depth*N+C+1)*emb_dim
        for l in range(depth):
            # attn
            num_heads = config["head"][l]
            head_dim = config["head_dim"][l] if "head_dim" in config.keys() else 0
            qk_dim = config["qk"][l] if "qk" in config.keys() else head_dim
            v_dim = config["v"][l] if "v" in config.keys() else head_dim
            flops += (2*N*emb_dim+N*N)*(qk_dim+v_dim)*num_heads
            # mlp
            mlp_dim = config["mlp"][l]
            flops += 2*N*emb_dim*mlp_dim

    gflops = flops/1e9
    return gflops

def get_mparam(model):
    return sum(p.numel() for p in model.parameters())/1e6

@torch.no_grad()
def profiling(
        args,model,
        input_size=(256,3,224,224),
        warm_steps=3,run_steps=10,
    ):
    print(f"Device: {args.device}")
    model.to(args.device)
    x = torch.randn(input_size).to(args.device)
    print(f"Warming up model ({warm_steps} runs)...")
    for i in tqdm(range(warm_steps)):
        model(x)
        if args.device == "cuda":
            torch.cuda.synchronize()
    
    peak_memories = []  # GB
    times = []  # ms
    pbar = tqdm(range(run_steps))
    for i in pbar:
        torch.cuda.synchronize()
        start_time = perf_counter()
        model(x)
        torch.cuda.synchronize()
        end_time = perf_counter()
        inference_time = (end_time - start_time)*1000
        times.append(inference_time)

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        peak_memories.append(peak_mem)

        pbar.set_postfix({
            "inference time (ms)": f"{inference_time:.2f}",
            "mem (GB)": f"{peak_mem:.2f}"
        })
    
    avg_inference_time = sum(times) / len(times)
    avg_peak_mem = sum(peak_memories) / len(peak_memories)

    inference_time = round(avg_inference_time,2)
    throughput = round(x.size(0) / (avg_inference_time) * 1000,2)
    peak_mem = round(avg_peak_mem,2)
    gflops = get_gflops(model)
    mparam = get_mparam(model)

    profiling_dict = {
        "inference time (ms)": inference_time,
        "throughput (image/s)": throughput,
        "peak_memory (GB)": peak_mem,
        "#Params (M)": round(mparam,2),
        "FLOPs (G)": round(gflops,2),
    }

    return profiling_dict

def reinit_weights(m: nn.Module):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()
    else:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def rate2step(rate):
    return int(6250*rate**2+1250*rate)

def get_post_training_memory(model,batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 清理缓存，避免之前的显存干扰
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 构造输入 (batch_size=256, 3, 224, 224)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)

    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    max_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"Maximum memory usage during training: {max_mem:.2f} GB")