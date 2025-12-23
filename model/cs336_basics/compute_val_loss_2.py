import argparse
import importlib
import os
import numpy as np
import torch

from cs336_basics import data_loader, cross_entropy_loss


def create_obj(cfg, field, addtl_params=None):
    class_path = cfg[field]["name"]
    cls_params = cfg[field]["params"]
    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    if addtl_params is not None:
        cls_params["params"] = addtl_params
    cls = getattr(mod, class_name)
    return cls(**cls_params)


def load_model_from_checkpoint(ckpt_path, device):
    # Load checkpoint
    obj = torch.load(ckpt_path, map_location=device)

    # Get config from the checkpoint to guarantee a match
    if isinstance(obj, dict) and "config" in obj:
        cfg = obj["config"]
    else:
        raise ValueError("Checkpoint does not contain a 'config' field")

    # Build model from this config
    LM = create_obj(cfg, field="model")
    LM = LM.to(device)
    LM.device = device  # ensure downstream modules use the actual runtime device

    # Extract and clean state dict
    state = obj.get("model_state", obj) if isinstance(obj, dict) else obj
    new_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        new_state[k] = v
    state = new_state

    missing, unexpected = LM.load_state_dict(state, strict=False)
    print(f"[load] missing: {missing} unexpected: {unexpected}")

    LM.eval()
    return LM, cfg


@torch.inference_mode()
def compute_validation_loss_for_checkpoint(ckpt_path):
    # ---- Device selection similar to training ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    LM, cfg = load_model_from_checkpoint(ckpt_path, device)

    # AMP / dtype like training
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        autocast_dtype = torch.bfloat16
        amp_flag = True
    elif device.type == "mps":
        autocast_dtype = torch.float16
        amp_flag = True
    else:
        autocast_dtype = torch.float32
        amp_flag = False

    # Model / validation params
    B = cfg["training"]["params"]["batch_size"]
    T = cfg["model"]["params"]["context_length"]
    theta = cfg["training"]["params"]["theta"]

    val_path = cfg["validation"]["data"]["path"]
    val_tag = cfg["validation"]["data"]["tag"]
    validation_num_shards = cfg["validation"]["data"]["num_shards"]
    validation_num_tries = cfg["validation"]["params"]["num_tries"]

    # Load validation memmaps like in training
    val_memmap_data = []
    for d in range(validation_num_shards):
        file_pattern = os.path.join(val_path, f"{val_tag}{d}.bin")
        data_rb = np.memmap(file_pattern, dtype=np.uint16, mode="r")
        val_memmap_data.append(data_rb)

    ce_loss = []
    LM.eval()
    for b_id in range(validation_num_shards):
        for b in range(validation_num_tries):
            tokenized_data = val_memmap_data[b_id]
            x, target = data_loader.data_loader(
                tokenized_data,
                batch_size=B,
                context_length=T,
                device_type=device.type,
            )

            x = x.to(device=device)
            target = target.to(device=device)
            token_pos = torch.arange(T, device=device)

            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=amp_flag
            ):
                logits = LM.tranform_lm_model(
                    x,
                    rope_theta=theta,
                    token_positions=token_pos[: x.size(1)],
                    max_seq_len=x.size(1),
                )
                loss = cross_entropy_loss.cross_entropy_loss(logits, target)

            ce_loss.append(loss.item())

    final_val_loss = sum(ce_loss) / len(ce_loss)
    print(f"Final validation loss for {ckpt_path}: {final_val_loss:.4f}")
    return final_val_loss


def get_args():
    p = argparse.ArgumentParser(
        description="Compute validation loss for a single checkpoint."
    )
    p.add_argument("ckpt_path", help="Path to a .pt checkpoint file")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    compute_validation_loss_for_checkpoint(args.ckpt_path)
