import numpy as np
import torch


def data_loader(x, batch_size, context_length, device_type=None):
    """
    data_loader takes a numpy array x (integer array with token IDs), a
    batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns
    a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both tensors
    should have shape (batch_size, context_length) containing token IDs, and both should be
    placed on the requested device

    Args :
        x - input array of tokens ( np.array)
        batch_size -size of batch
        context_length - length of context to split the corpus
        device_type - location in which the data has to be loaded and split

        Returns :
            y - sampled input sequences ( tensor: batch_size x context_length)
            t - targets ( tensor:  batch_sizr x context_length)
    """
    # device
    if device_type is None or device_type == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "mps") and torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    # Convert x to numpy, put it in device and ensure they are
    # in contiguous memory
    x = (
        torch.as_tensor(np.array(x, copy=True), dtype=torch.long)
        .to(device)
        .contiguous()
    )
    # generate 32 different starts from all possible indices.
    # Available indices is len(x) - context_length
    strt_idx = torch.randperm(len(x) - context_length, device=device)
    strt_idx = strt_idx[:batch_size]
    tgt_strt = strt_idx + 1

    context_array = torch.arange(context_length, device=device)
    slice_idx_d = strt_idx[:, None] + context_array[None, :]
    slice_idx_t = tgt_strt[:, None] + context_array[None, :]

    y = x[slice_idx_d].to(device)
    t = x[slice_idx_t].to(device)
    return y, t


if __name__ == "__main__":
    data_loader(x=np.array(range(1000)), batch_size=100, context_length=256)
