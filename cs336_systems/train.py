import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
import timeit 
from cs336_basics.dataloader import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.transformer import TransformerLM, cross_entropy, scaled_dot_product_attn
from cs336_basics.optimizer import AdamW, gradient_clipping, lr_schedule 
from cs336_basics.tokenizer import BPETokenizer
from cs336_systems.utils import parse_args, set_seed, load_data, evaluate, decoding, sync, benchmarking_print, warm_up, pack_hook, unpack_hook, stats

def attention_benchmark():
    device = "cuda" if torch.cuda.is_available() else \
            "mps" if torch.backends.mps.is_available() else "cpu"
    for d_model in [16, 32, 64, 128]:
        for seq_len in [256, 1024, 4096, 8192, 16384]:
            Q = torch.randn(8, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(8, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(8, seq_len, d_model, device=device, requires_grad=True)

            for _ in range(10):  # 预热
                out = scaled_dot_product_attn(Q, K, V)
                sync(device)
            
            mem = torch.cuda.memory_allocated()
            
            fwd_elapsed = 0
            bwd_elapsed = 0
            for _ in range(100):
                fwd_start = timeit.default_timer()
                out = scaled_dot_product_attn(Q, K, V)
                sync(device)
                fwd_end = timeit.default_timer()
                fwd_elapsed += fwd_end - fwd_start 
                loss = out.sum()
                loss.backward()
                sync(device)
                bwd_elapsed += timeit.default_timer() - fwd_end
                Q.grad = K.grad = V.grad = None
            print(f"d_model {d_model} seq_len {seq_len} forward 100 times: {fwd_elapsed:.4f} seconds")
            print(f"d_model {d_model} seq_len {seq_len} backward 100 times: {bwd_elapsed:.4f} seconds")
                
    

def main():
    start_time = timeit.default_timer()
    args = parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else \
            "mps" if torch.backends.mps.is_available() else "cpu"
    
    if args.bench is not None:
        train_data = np.random.randint(0, args.vocab_size, args.vocab_size)
    else:
        train_data = load_data(args.train_data)
        val_data = load_data(args.val_data)

    ctx = torch.autocast(device, dtype=torch.float16) if args.use_amp else nullcontext()
    
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    ckpt_dir = Path("checkpoints") / run_name
    # ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = TransformerLM(
        vocab_size=args.vocab_size, 
        context_length=args.context_length, 
        num_layers=args.num_layers, 
        d_model=args.d_model, 
        num_heads=args.num_heads, 
        d_ff=args.d_ff, 
        rope_theta=args.rope_theta,
        use_checkpoint=True
    ).to(device)
    model = torch.compile(model)
    
    optimizer = AdamW(
        params=model.parameters(), 
        betas=args.betas, 
        weight_decay=args.weight_decay, 
        lr=args.lr, 
        eps=args.eps
    )
    
    start_step = 0
    if args.resume_from:
        ckpt_path = ckpt_dir / args.resume_from
        start_step = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resume from step {start_step}")

    model.train()

    # benchmarking 相关的变量
    benchmarking_start = 0
    elapsed = 0
    time_steps = []

    print(f"training on {device} with batch size {args.batch_size}, context length {args.context_length}, for {args.num_steps} steps...")
    if args.bench:
        print(f"Running benchmarking for {args.bench_steps} steps after {args.warmup} warmup steps...")
   
    for step in range(start_step, args.num_steps):
        if args.bench is not None and step < args.warmup:
            # 预热阶段，不计时
            warm_up(model, optimizer, train_data, ctx, args, device)
            continue
        
        t_step_start = timeit.default_timer()

        lr = lr_schedule(step, args.lr_max, args.final_lr, args.warmup_steps, args.cos_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        if args.bench is not None:
            benchmarking_start = timeit.default_timer() 
            
        # with nvtx.range("forward"):
        #     logits = model(x)
        hook_ctx = (torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook) if step == args.warmup else nullcontext())
        with hook_ctx, ctx:
            logits = model(x)
        
        if args.bench == "forward":
            sync(device)
            time_steps.append(timeit.default_timer() - benchmarking_start)
        
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()

        # with nvtx.range("backward"):
        #     loss.backward()
        with ctx:
            loss.backward()
        
        if args.bench == "backward":
            sync(device)
            time_steps.append(timeit.default_timer() - benchmarking_start)
        
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
    
        # with nvtx.range("optimizer"):
        #     optimizer.step()
        with ctx:
            optimizer.step()
        
        if args.bench == "optim":
            sync(device)
            time_steps.append(timeit.default_timer() - benchmarking_start)

        if not args.bench:
            
            if step % args.log_interval == 0:
                elapsed = timeit.default_timer() - t_step_start
                tokens_per_sec = args.batch_size * args.context_length / elapsed
                print(f"step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}  tok/s {tokens_per_sec:,.0f}")

            if step % args.val_interval == 0 and step > 0:
                val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device, args.n_eval_batches)
                print(f"steps {step} val loss {val_loss:.4f}")

            if step % args.ckpt_interval == 0 and step > 0:
                ckpt_path = ckpt_dir / f"{step}"
                save_checkpoint(model, optimizer, step, ckpt_path)

    benchmarking_print(args, time_steps) 
    
    # tokenizer = BPETokenizer.from_files(args.vocab_path, args.merges_path)

    if args.generate is not None:
        decoding(model, tokenizer, args.generate, args.max_token_len, args.temperature, args.p, device=device, context_length=50)

    # save_checkpoint(model, optimizer, args.num_steps, ckpt_dir / "final.pt")
    
    end_time = timeit.default_timer()
    print(f"Total training time: {(end_time - start_time):.2f} seconds")
    print(f"Total size of saved tensors in Model: {stats["total_size_bytes"] / (1024**2):.2f} MiB")

if __name__ == "__main__":
    # main()
    attention_benchmark()

