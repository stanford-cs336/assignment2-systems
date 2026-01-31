from copy import deepcopy
import os
from einops import rearrange
from tests.common import (
    ToyModel,
    _cleanup_process_group,
    _setup_process_group,
)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


def run_naive_ddp(rank, world_size, backend, global_x, global_y, initial_weights):
  """Executes Naïve Distributed Data Parallel training on a specific rank.

  Args:
      rank: The process ID (0, 1, ...).
      world_size: Total number of processes.
      backend: 'gloo' (CPU) or 'nccl' (GPU). global_x, global_y: The full
        dataset tensors (same on all ranks).
  """
  # init processes, rank
  device = _setup_process_group(rank, world_size, backend)
  model = ToyModel().to(device)
  model.load_state_dict(initial_weights)
  ddp_optimizer = optim.SGD(model.parameters(), lr=0.1)
  loss_func = nn.MSELoss()

  # broadcast weights
  for param in model.parameters():
    # 'src=0' means Rank 0 sends the data, everyone else receives it
    dist.broadcast(param.data, src=0)
  # data sharding
  local_x = rearrange(global_x, "(w b) ... -> w b ...", w=world_size)[rank].to(
      device
  )
  local_y = rearrange(global_y, "(w b) ... -> w b ...", w=world_size)[rank].to(
      device
  )

  for _ in range(5):
    # forward step
    logits = model(local_x)
    # backward step
    loss = loss_func(logits, local_y)
    ddp_optimizer.zero_grad()
    loss.backward()

    # Sync gradients across workers
    for param in model.parameters():
      if param.grad is None:
          continue
      # We use ReduceOp.AVG to get the average gradient across all ranks
      dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

    ddp_optimizer.step()
    # all reduce and average on the gradient

  if rank == 0:
        torch.save(model.state_dict(), "ddp_final_model.pt")
  _cleanup_process_group()


def run_single_process(global_x, global_y, initial_weights):
    """
    Runs standard, non-distributed training for verification.
    This serves as the 'Ground Truth' to check your DDP implementation against.
    """
    # 1. Setup Model with Identical Weights
    device = "cuda"
    global_x = global_x.to(device)
    global_y = global_y.to(device)
    model = ToyModel().to(device)
    model.load_state_dict(initial_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_func = nn.MSELoss()

    # 2. Training Loop
    # Note: No data sharding! The single process sees the FULL global batch.
    for _ in range(5):
        logits = model(global_x)
        loss = loss_func(logits, global_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model.state_dict()

if __name__ == "__main__":
  # Example trigger for CPU testing
  # For benchmarking, you will use backend='nccl' and world_size=2 on GPUs
  world_size = 2
  # Create random data (Batch size 32, Input 10, Output 5)
  global_x = torch.randn(32, 10)
  global_y = torch.randn(32, 5)
  temp_model = ToyModel()
  initial_weights = deepcopy(temp_model.state_dict())
  mp.spawn(
      run_naive_ddp,
      args=(world_size, "nccl", global_x, global_y,initial_weights),
      nprocs=world_size,
      join=True,
  )
  print("Training complete. Loading model in main process...")
  final_model = ToyModel()
  final_model.load_state_dict(torch.load("ddp_final_model.pt"))
  print("Model loaded successfully!")

  single_proc_weights = run_single_process(global_x, global_y, initial_weights)
  ddp_weights = final_model.state_dict()

  match = True
  for k in single_proc_weights:
    # We use a small tolerance (atol) because floating point math 
    # can differ slightly between CPU (single proc) and GPU (DDP reduction)
    if not torch.allclose(single_proc_weights[k].cpu(), ddp_weights[k].cpu(), atol=1e-5):
        print(f"❌ Mismatch in {k}")
        match = False
  
  if match:
      print("✅ SUCCESS: DDP matches Single Process exactly!")
  else:
      print("❌ FAILED: Weights do not match.")
