import triton
import math
import torch
import timeit
from flash_attention_2_pytroch import FlashAttention2Pytorch
from flash_attention_2_triton import FlashAttention2Triton
# For each precision:
#     For each embedding dimension:
#         For each sequence length:

#             Generate Q, K, V
#             Configure batch size = 1
#             Configure causal = True
#             Construct both implementations
#             Configure any tile sizes

#             Benchmark:
#                 Forward
#                 Backward
#                 Forward + Backward


# def FullPassTriton(Q, K, V, is_casual=True):
#     y = FlashAttention2Triton.forward(Q, K, V, is_causal=is_casual)
#     y.backward()

def main():
    name = input("Forward: 1, Backward: 2 or Full: 3? ")
    print(f"Your option is: {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # precision = [torch.bfloat16, torch.float32]
    precision = [torch.float32]
    embedding_dimensions = [16]
    sequence_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # embedding_dimensions = [16, 32, 64, 128]
    # sequence_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    for typ in precision:
        for emb_dim in embedding_dimensions:
            for seq_length in sequence_lengths:
                print("#############################")
                print(f"Type: {typ}, emb_dim: {emb_dim}, seq_length: {seq_length}")
                dtype = typ
                shape = (1, seq_length, emb_dim)
                q_tensor = torch.randn(shape, dtype=dtype, device=device)
                k_tensor = torch.randn(shape, dtype=dtype, device=device)
                v_tensor = torch.randn(shape, dtype=dtype, device=device)

                is_casual = True

                def ForwardPytorch():
                    FlashAttention2Pytorch.apply(q_tensor, k_tensor, v_tensor, is_casual)

                def ForwardTriton():
                    FlashAttention2Triton.apply(q_tensor, k_tensor, v_tensor, is_casual)

                # 2. Run Forward Pass ONCE outside the benchmarker
                y_pytorch = FlashAttention2Pytorch.apply(q_tensor, k_tensor, v_tensor, is_casual)
                grad_outputs_pytorch = torch.ones_like(y_pytorch)

                y_triton = FlashAttention2Triton.apply(q_tensor, k_tensor, v_tensor, is_casual)
                grad_outputs_triton = torch.ones_like(y_triton)

                def BenchmarkBackwardPytorch():
                    torch.autograd.backward(y_pytorch, grad_outputs_pytorch, retain_graph=True)

                def BenchmarkBackwardTriton():
                    torch.autograd.backward(y_triton, grad_outputs_triton, retain_graph=True)
                    
                def FullPassPytorch():
                    y = FlashAttention2Pytorch.apply(q_tensor, k_tensor, v_tensor, is_casual)
                    y.backward()
                def FullPassTriton():
                    y = FlashAttention2Triton.apply(q_tensor, k_tensor, v_tensor, is_casual)
                    y.backward()

                warm_up = 5
                rep = 10
                grad_to_none = None
                return_mode = "mean"
                if name == "1":
                    f_pytorch = triton.testing.do_bench(fn=ForwardPytorch, rep=rep, warmup=warm_up, 
                                            grad_to_none=grad_to_none, return_mode=return_mode)
                    f_triton = triton.testing.do_bench(fn=ForwardTriton, rep=rep, warmup=warm_up, 
                                                            grad_to_none=grad_to_none, return_mode=return_mode)
                    print(f"f_pytorch: {f_pytorch} and f_triton: {f_triton}")
                # triton.testing.do_bench(fn=BenchmarkBackwardPytorch, rep=rep, warmup=warm_up, 
                #                                         grad_to_none=grad_to_none, return_mode=return_mode)
                # triton.testing.do_bench(fn=BenchmarkBackwardTriton, rep=rep, warmup=warm_up, grad_to_none=grad_to_none, return_mode=return_mode)
                if name == "3":
                    fnb_pytorch = triton.testing.do_bench(fn=FullPassPytorch, rep=rep, warmup=warm_up, grad_to_none=grad_to_none, return_mode=return_mode)
                    fnb_triton = triton.testing.do_bench(fn=FullPassTriton, rep=rep, warmup=warm_up, 
                                                            grad_to_none=grad_to_none, return_mode=return_mode)

                    print(f"f_pytorch: {fnb_pytorch} and f_triton: {fnb_triton}")
                
                



if __name__ == "__main__":
    main()