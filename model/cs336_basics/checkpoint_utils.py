# cs336_basics/checkpoint_utils.py
import torch


# -------------------------
# Save in your canonical format
# -------------------------
def save_checkpoint(model, optimizer, iteration, out):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "lr_iterations": int(iteration),
    }
    torch.save(payload, out)


# -------------------------
# Helpers
# -------------------------
def _pick_device(explicit=None):
    if explicit is not None:
        return torch.device(explicit)
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_model_opt_iters(obj):
    """
    Accepts any of:
      - {"model_state": ..., "optimizer_state": ..., "lr_iterations": ...}
      - {"model_state_dict": ...}
      - {"state_dict": ...}
      - a raw model state_dict (OrderedDict of tensors)
    Returns: (model_state_dict, optimizer_state_dict_or_None, lr_iters_or_None)
    """
    # canonical form
    if isinstance(obj, dict) and "model_state" in obj:
        return (
            obj["model_state"],
            obj.get("optimizer_state", None),
            obj.get("lr_iterations", None),
        )

    # common variants
    for k in ("model_state_dict", "state_dict"):
        if isinstance(obj, dict) and k in obj:
            return (
                obj[k],
                obj.get("optimizer_state", None),
                obj.get("lr_iterations", None),
            )

    # raw state_dict
    if isinstance(obj, dict):
        # Heuristic: look like a state dict if tensor leaves
        has_tensor_leaf = any(torch.is_tensor(v) for v in obj.values())
        if has_tensor_leaf:
            return obj, None, None

    raise ValueError(
        "Unrecognized checkpoint format: expected keys like "
        "'model_state', 'model_state_dict', 'state_dict', or a raw state_dict."
    )


# -------------------------
# Robust loader
# -------------------------
def load_checkpoint(src, model, optimizer=None, device=None, *, strict: bool = False):
    """
    Loads a checkpoint from path/file-like `src` and restores:
      - model weights (always)
      - optimizer state (if provided AND present)
    It accepts multiple formats (your canonical format, PyTorch-style dicts, or raw state_dicts).
    Tensors are mapped to `device` (or auto-detected) and cast to the model's dtype.

    Returns:
      lr_iterations (int | None)
    """
    target_device = _pick_device(device)
    chk = torch.load(src, map_location=target_device)

    model_sd, opt_sd, iters = _extract_model_opt_iters(chk)

    # Match model dtype & current device
    target_dtype = next(model.parameters()).dtype
    target_device = next(model.parameters()).device

    casted = {}
    for k, v in model_sd.items():
        if torch.is_tensor(v):
            casted[k] = v.to(device=target_device, dtype=target_dtype)
        else:
            casted[k] = v

    missing, unexpected = model.load_state_dict(casted, strict=strict)
    if missing or unexpected:
        print("load_state_dict  missing:", missing, "  unexpected:", unexpected)

    if optimizer is not None and opt_sd is not None:
        # Optimizer tensors should already be on CPU in state dict; PyTorch will move as needed
        optimizer.load_state_dict(opt_sd)

    return iters
