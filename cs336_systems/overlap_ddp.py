import torch
import torch.nn as nn
import torch.distributed as dist

class OverlapDDP(nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.module = model
        self.handles = []
        
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
            if p.requires_grad:
                # Use a closure or partial to capture the specific parameter 'p'
                p.register_post_accumulate_grad_hook(self._make_hook(p))
                
    def _make_hook(self, p):
        def hook(param):
            if param.grad is None:
                return
            
            # 1. Manually divide by world size to simulate averaging
            #    We do this in-place before sending.
            #    This makes it compatible with Gloo (CPU) which lacks ReduceOp.AVG
            world_size = dist.get_world_size()
            param.grad.data /= world_size
            
            # 2. Fire async communication using SUM
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            
            self.handles.append(handle)
        return hook

    def wait_for_all(self):
        for h in self.handles:
            h.wait()
        self.handles = []

    def forward(self, *inputs, **kwargs):
        """
        Standard forward pass. 
        Since this is a wrapper, we just pass inputs to the underlying model.
        """
        return self.model(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Waits for all asynchronous communication to finish.
        MUST be called after loss.backward() and before optimizer.step().
        """
        for handle in self.handles:
            handle.wait()
        self.handles.clear()