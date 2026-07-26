import torch
import math

class FlashAttention2Pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V: torch.tensor, is_causal=False):
        _bq = 16
        _bk = 16
        final_o_list = []
        final_l_list = []
        attention_state = {}
        if K.shape != V.shape:
            raise "K and V shape should match."

        for chunk_Q, chunk_K, chunk_V in zip(Q, K, V):
            nq, d = chunk_Q.shape
            nk, d = chunk_K.shape
            scale_d = 1.0/math.sqrt(d)
            # q_num_chunks = nq // _bq
            # k_num_chunks = nk // _bk

            q_chunks = torch.split(chunk_Q, _bq, dim=0)
            k_chunks = torch.split(chunk_K, _bk, dim=0)
            v_chunks = torch.split(chunk_V, _bk, dim=0)

            counter_i = -1
            for q_chunk in q_chunks:
                counter_i+=1
                # load Q from HBM(pretending to load as this is pytorch) 

                attention_state[f"O{counter_i}_0"] = torch.zeros(_bq, d)
                attention_state[f"l{counter_i}_0"] = torch.zeros(_bq)
                attention_state[f"m{counter_i}_0"] = torch.full((_bq,), float('-inf'))
                counter_j = -1
                for k_chunk, v_chunk in zip(k_chunks, v_chunks):
                    counter_j+=1
                    if counter_j == 0:
                        prev_O_key = f"O{counter_i}_0"
                        prev_l_key = f"l{counter_i}_0"
                        prev_m_key = f"m{counter_i}_0"
                    else:
                        prev_O_key = f"O{counter_i}_{counter_j-1}"
                        prev_l_key = f"l{counter_i}_{counter_j-1}"
                        prev_m_key = f"m{counter_i}_{counter_j-1}"

                    O_prev = attention_state[prev_O_key]
                    l_prev = attention_state[prev_l_key]
                    m_prev = attention_state[prev_m_key]
                    attention_state[f"S{counter_i}_{counter_j}"] = torch.matmul(q_chunk, k_chunk.T)*scale_d
                    attention_state[f"m{counter_i}_{counter_j}"] = torch.max(m_prev, torch.max(attention_state[f"S{counter_i}_{counter_j}"], dim=1).values)
                    attention_state[f"P{counter_i}_{counter_j}"] = torch.exp(attention_state[f"S{counter_i}_{counter_j}"]-attention_state[f"m{counter_i}_{counter_j}"][:, None])
                    attention_state[f"l{counter_i}_{counter_j}"] = torch.exp(m_prev-attention_state[f"m{counter_i}_{counter_j}"])*l_prev + torch.sum(attention_state[f"P{counter_i}_{counter_j}"], dim=1)
                                            
                    scale_vector = torch.exp(m_prev - attention_state[f"m{counter_i}_{counter_j}"])
                    scaled_O_prev = scale_vector[:, None] * O_prev
                    attention_state[f"O{counter_i}_{counter_j}"] = scaled_O_prev + torch.matmul(attention_state[f"P{counter_i}_{counter_j}"], v_chunk)
                    

                attention_state[f"O{counter_i}"] = attention_state[f"O{counter_i}_{counter_j}"] / attention_state[f"l{counter_i}_{counter_j}"] [:, None]
                attention_state[f"l{counter_i}"] = attention_state[f"m{counter_i}_{counter_j}"] + torch.log(attention_state[f"l{counter_i}_{counter_j}"])
            
            o_list = []
            l_list = []

            for i in range(counter_i+1):
                o_list.append(attention_state[f"O{i}"])
                l_list.append(attention_state[f"l{i}"])

            final_o_list.append(torch.cat(o_list, dim=0))
            final_l_list.append(torch.cat(l_list, dim=0))
        

        print(f"Q shape: {Q.shape}")
        O = torch.stack(final_o_list, dim = 0)
        L = torch.stack(final_l_list, dim = 0)
        print(f"Q shape: {Q.shape}")
        print(f"O shape: {O.shape}")
        print(L.shape)
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        Bq, Sq, Dq = Q.shape
        D = torch.sum(O*grad_output, dim=-1)
        scale_d = 1.0/math.sqrt(Dq)
        S = torch.matmul(Q, K.transpose(-1, -2))*scale_d
        P = torch.exp(S - L[:, :, None])
        dV = torch.matmul(P.transpose(-1, -2), grad_output)
        dP = torch.matmul(grad_output, V.transpose(-1, -2)) 
        dS = P * (dP - D[:, :, None])
        dQ = torch.matmul(dS, K)*scale_d
        dK = torch.matmul(dS.transpose(-1, -2), Q)*scale_d
        return dQ, dK, dV, None
