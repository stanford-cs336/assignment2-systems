import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange

# --- LOCAL IMPORTS ---
# Ensure your PYTHONPATH is set correctly to find this
from tests.common import _cleanup_process_group, _setup_process_group
try:
    from cs336_basics.cs336_basics.model import BasicsTransformerLM
except ImportError:
    print("❌ Error: Could not import BasicsTransformerLM. Check PYTHONPATH.")

# --- XL CONFIGURATION ---
XL_CONFIG = {
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,
    "vocab_size": 10000,
    "context_length": 128,  # Kept small for memory safety on standard GPUs
}

def run_naive_ddp(rank, world_size, backend, global_input_ids, global_target_ids):
    """Executes Naïve DDP Benchmarking with the XL Model."""
    
    # 1. SETUP
    device = _setup_process_group(rank, world_size, backend)
    
    # Initialize the XL Model
    # Note: We disable gradients for inputs to save memory/compute if not needed
    model = BasicsTransformerLM(
        vocab_size=XL_CONFIG["vocab_size"],
        context_length=XL_CONFIG["context_length"],
        d_model=XL_CONFIG["d_model"],
        d_ff=XL_CONFIG["d_ff"],
        num_layers=XL_CONFIG["num_layers"],
        num_heads=XL_CONFIG["num_heads"],
        rope_theta=10000.0,
    ).to(device)
    
    # Optimizer
    ddp_optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Helper for timing
    sync = lambda: torch.cuda.synchronize() if backend == "nccl" else None

    # 2. BROADCAST WEIGHTS (Critical for DDP correctness)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # 3. DATA SHARDING (Integer Tokens)
    # We split the batch dimension (0)
    local_input_ids = rearrange(global_input_ids, "(w b) ... -> w b ...", w=world_size)[rank].to(device)
    local_target_ids = rearrange(global_target_ids, "(w b) ... -> w b ...", w=world_size)[rank].to(device)

    local_comm_times = []
    local_compute_times = []

    # 4. BENCHMARK LOOP
    # We run a few warmup steps if possible, but for simplicity here we just run 5 measured
    print(f"Rank {rank} starting training loop...")
    
    for step in range(5):
        # --- A. COMPUTE START ---
        sync()
        t0 = time.perf_counter()
        
        # Forward
        logits = model(local_input_ids)
        
        # Reshape for CrossEntropy: (Batch * Seq, Vocab)
        loss = F.cross_entropy(
            logits.view(-1, XL_CONFIG["vocab_size"]), 
            local_target_ids.view(-1)
        )
        
        ddp_optimizer.zero_grad()
        loss.backward()
        
        sync()
        t1 = time.perf_counter()
        # --- A. COMPUTE END ---

        # --- B. COMMUNICATION START ---
        sync()
        t2 = time.perf_counter()
        
        for param in model.parameters():
            if param.grad is None:
                continue
            # Naive DDP: Send every tensor individually
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
            
        sync()
        t3 = time.perf_counter()
        # --- B. COMMUNICATION END ---

        ddp_optimizer.step()

        # Record metrics
        local_compute_times.append(t1 - t0)
        local_comm_times.append(t3 - t2)
        
        if rank == 0:
            print(f"Step {step} finished.")

    # 5. AGGREGATE STATS
    avg_compute = sum(local_compute_times) / len(local_compute_times)
    avg_comm = sum(local_comm_times) / len(local_comm_times)
    
    # Pack into tensor for gathering: [Rank, Compute, Comm]
    local_stats = torch.tensor([rank, avg_compute, avg_comm], device=device)
    gathered_stats = [torch.zeros_like(local_stats) for _ in range(world_size)]
    dist.all_gather(gathered_stats, local_stats)

    if rank == 0:
        print("\n=== XL MODEL BENCHMARK RESULTS ===")
        for t in gathered_stats:
            r_id = int(t[0].item())
            r_comp = t[1].item()
            r_comm = t[2].item()
            r_total = r_comp + r_comm
            
            print(
                f"Rank {r_id} | "
                f"Compute: {r_comp:.4f}s | "
                f"Comm: {r_comm:.4f}s | "
                f"Overhead: {r_comm/r_total:.1%}"
            )

    _cleanup_process_group()

if __name__ == "__main__":
    world_size = 2
    batch_size = 4 # As per assignment spec for XL model
    
    # Generate Random Integer Data (Token IDs)
    # Shape: (Batch, Context_Length)
    print("Generating XL Model Data...")
    global_input_ids = torch.randint(
        0, XL_CONFIG["vocab_size"], 
        (batch_size, XL_CONFIG["context_length"]), 
        dtype=torch.long
    )
    # Targets are usually shifted or same size, just using random for benchmarking
    global_target_ids = torch.randint(
        0, XL_CONFIG["vocab_size"], 
        (batch_size, XL_CONFIG["context_length"]), 
        dtype=torch.long
    )

    print("Spawning processes...")
    mp.spawn(
        run_naive_ddp,
        args=(world_size, "nccl", global_input_ids, global_target_ids),
        nprocs=world_size,
        join=True,
    )