
import torch
import numpy as np
import os
import math
from einops import einsum
from torch import Tensor
from jaxtyping import Float, Bool
import argparse
from cs336_basics.transformer import cross_entropy, softmax
from cs336_basics.dataloader import get_batch
from cs336_basics.optimizer import AdamW, gradient_clipping, lr_schedule 

# args 
def parse_args(): 
    parser = argparse.ArgumentParser()
    # 数据集路径
    parser.add_argument("--train_data", type=str, required=False)
    parser.add_argument("--val_data", type=str, required=False)
    
    # 模型超参：d_model, num_layers, num_heads, d_ff, vocab_size, context_length, rope_theta
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # 优化器超参：lr, betas, weight_decay, eps
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9,0.999])
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-8)
    
    # 训练超参：batch_size, num_steps, grad_clip
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # LR schedule：warmup_steps, cosine_decay_end (or final_lr)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--cos_steps", type=int, default=2500)
    parser.add_argument("--final_lr", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-2)
    
    # IO：train_data, val_data, ckpt_dir, log_interval, val_interval, ckpt_interval
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    
    # 其他：seed, device, resume_from
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--n_eval_batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--generate", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_token_len", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--merges_path", type=str, default=None)

    # benchmarking
    parser.add_argument("--bench", type=str, default=None, choices=["forward", "backward", "optim"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bench_steps", type=int, default=5) 
    
    parser.add_argument("--use_amp", action="store_true") 
    
    return parser.parse_args()

def set_seed(seed: int):
    torch.manual_seed(seed)

# file path
def load_data(path: str | os.PathLike):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    return data

@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, n_eval_batches):
    model.eval()
    total_loss = 0
    for _ in range(n_eval_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x) 
        loss = cross_entropy(logits, y)
        total_loss += loss.item() 
    model.train()
    return total_loss / n_eval_batches
     
@torch.no_grad()
def decoding(
    model, 
    tokenizer, 
    prompt: str, 
    max_token_len: int, 
    temperature: float, 
    p: float,
    device,
    context_length: int
):
    model.eval()

    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # 输入需要有 batch 维度

    # 尝试找 EOS token id
    try:
        eot_id = tokenizer.reversed_vocab[b"<|endoftext|>"]
    except KeyError:
        eot_id = None 

    generated_ids = []
    
    for _ in range(max_token_len):
        # 截断到 context_length，生产中是用 KV Cache
        if ids.shape[1] > context_length:
            input_ids = ids[:, -context_length:]
        else:
            input_ids = ids # (1, s, v)
            
        logits = model(input_ids)[0, -1, :]     # (v,)
        
        probs = softmax(logits, temperature=temperature, dim=-1)
        
        # top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True) 
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # 找到第一个 cumulative >= p 的位置，保留到这里
        # nucleus 包括这个位置（"smallest set such that sum >= p"）
        # 求第一个满足的下标，nonzero 返回非 0 的坐标
        # item() 会返回 Number 基类，不能直接用
        cutoff = (cumulative >= p).nonzero()[0].item()          
        nucleus_probs = sorted_probs[:cutoff + 1]
        nucleus_indices = sorted_indices[:cutoff + 1]
        
        nucleus_probs= nucleus_probs / nucleus_probs.sum()
        sampled_pos = torch.multinomial(nucleus_probs, num_samples=1)
        next_id = nucleus_indices[sampled_pos].item()

        # endoftext 切断
        if eot_id is not None and next_id == eot_id:
            break
        
        # 不要把生成的 token 转换成 str 再拼回去重新 encode，效率非常低下
        # 直接把生成的 token 拼回 encode 后的 ids
        generated_ids.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1) # (1, s)，沿着 seq_len 维度拼接
        
    output_text = tokenizer.decode(generated_ids)
    print("Prompt:", prompt)
    print("Generation:", output_text)
    
    model.train()
    return output_text
    
        
# helpers for benchmarking, 同步设备，确保计时准确
def sync(device: str):
    if device == "mps":
        torch.mps.synchronize() 
    elif device == "cuda":
        torch.cuda.synchronize()
    else:
        return  

def benchmarking_print(args, time_steps):
    if args.bench == "forward":
        print(f"Forward average timing: {np.average(time_steps):.4f} seconds per step, standard deviation: {np.std(time_steps):.4f} seconds")
    if args.bench == "backward":
        print(f"Forward and backward without counting optimizer average timing {np.average(time_steps):.4f} seconds, standard deviation: {np.std(time_steps):.4f} seconds")
    if args.bench == "optim":
        print(f"Forward and backward with counting optimizer average timing {np.average(time_steps):.4f} seconds, standard deviation: {np.std(time_steps):.4f} seconds")

import torch.cuda.nvtx as nvtx
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"], 
    K: Float[Tensor, "... keys d_k"], 
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:
    with nvtx.range("computing attention scores"):
        # compute attention scores between Q and K
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    with nvtx.range("computing softmax"):
        # compute softmax of attention scores
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    with nvtx.range("final matmul"):
        # compute output projection
        out = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return out

# cs336_basics.transformer.scaled_dot_product_attn = annotated_scaled_dot_product_attention

def warm_up(model, optimizer, train_data, ctx, args, device):
    x, y = get_batch(train_data, args.batch_size, args.context_length, device)  
    lr = lr_schedule(0, args.lr_max, args.final_lr, args.warmup_steps, args.cos_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    with ctx:
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
    optimizer.step()
    if args.grad_clip > 0:
        gradient_clipping(model.parameters(), args.grad_clip)

stats = {"total_size_bytes": 0}
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter): # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    stats["total_size_bytes"] += t.numel() * t.element_size()
    # print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    # print(f"Loading residual: {shape:} {dtype:} {grad_fn:}")
    return t
