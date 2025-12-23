import argparse
import importlib
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

from cs336_basics import data_loader, cross_entropy_loss


def create_obj(cfg, field, addtl_params=None):
    """
    Create an object from cfg[field]["name"] with kwargs in cfg[field]["params"].
    """
    class_path = cfg[field]["name"]
    cls_params = cfg[field]["params"]

    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)

    if addtl_params is not None:
        cls_params["params"] = addtl_params

    cls = getattr(mod, class_name)
    return cls(**cls_params)


def load_model_from_checkpoint(ckpt_path):
    """
    Load LM + cfg from a checkpoint, handling _orig_mod./module. prefixes
    and using the model's own device (cfg['model']['params']['device']).
    """
    # Load checkpoint onto CPU first
    obj = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(obj, dict) or "config" not in obj:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not contain a 'config' field. "
            "Did you save the training cfg into the checkpoint?"
        )

    cfg = obj["config"]

    # Build model from config (this sets LM.device internally)
    LM = create_obj(cfg, field="model")

    # Use the model's own device from the config
    device = LM.device  # your TransformerLM exposes this
    print(f"[build {os.path.basename(ckpt_path)}] LM.device = {device}")

    # Move LM to its own device (usually no-op but safe)
    LM = LM.to(device)

    # Clean state dict keys (_orig_mod., module.)
    state = obj.get("model_state", obj)
    new_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        new_state[k] = v

    missing, unexpected = LM.load_state_dict(new_state, strict=False)
    if missing:
        # When the checkpoint was produced from a compiled/DataParallel model the raw
        # state dict keys are prefixed (e.g. `_orig_mod.`). If we forget to strip them,
        # nothing gets loaded and evaluation will run with random weights, which shows
        # up as a CE loss near log(|V|). Make this obvious instead of silently
        # returning garbage numbers.
        raise ValueError(
            f"{ckpt_path} is missing parameters after load: {missing}. "
            "Double-check the prefix-stripping logic in load_model_from_checkpoint."
        )
    if unexpected:
        print(
            f"[load {os.path.basename(ckpt_path)}] unexpected entries in checkpoint: {unexpected}"
        )
    else:
        print(f"[load {os.path.basename(ckpt_path)}] checkpoint restored successfully.")

    LM.eval()
    return LM, cfg, device


@torch.inference_mode()
def compute_validation_loss_for_checkpoint(ckpt_path):
    """
    Compute average validation CE loss for a single checkpoint.
    Uses the validation config stored inside the checkpoint.
    """
    LM, cfg, device = load_model_from_checkpoint(ckpt_path)

    # AMP / dtype logic mirroring your training code, but based on LM.device
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
    num_shards = cfg["validation"]["data"]["num_shards"]
    num_tries = cfg["validation"]["params"]["num_tries"]

    # Load validation memmaps exactly like training
    val_memmap_data = []
    for d in range(num_shards):
        file_pattern = os.path.join(val_path, f"{val_tag}{d}.bin")
        if not os.path.exists(file_pattern):
            raise FileNotFoundError(f"Validation shard not found: {file_pattern}")
        data_rb = np.memmap(file_pattern, dtype=np.uint16, mode="r")
        val_memmap_data.append(data_rb)

    ce_loss = []
    LM.eval()

    print(
        f"[val] Using device={device} | B={B} T={T} | shards={num_shards} tries={num_tries}"
    )

    for shard_id in range(num_shards):
        for t in range(num_tries):
            tokenized_data = val_memmap_data[shard_id]
            x, target = data_loader.data_loader(
                tokenized_data,
                batch_size=B,
                context_length=T,
                device_type=device.type,  # <<< keep x/target on same device as LM
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


def extract_step_from_name(filename: str):
    """
    Try to extract step number from names like 'iter_3600.pt'.
    If it fails, return None.
    """
    m = re.search(r"iter_(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate one or many checkpoints and (optionally) plot a sweep."
    )
    parser.add_argument(
        "expt_path",
        help="Path to a checkpoint directory OR a single .pt checkpoint file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set and expt_path is a directory, plot val loss vs step.",
    )

    args = parser.parse_args()
    path = args.expt_path

    # Case 1: user passed a single .pt file
    if os.path.isfile(path) and path.endswith(".pt"):
        compute_validation_loss_for_checkpoint(path)
        return

    # Case 2: directory of checkpoints
    if not os.path.isdir(path):
        raise ValueError(f"{path} is neither a .pt file nor a directory")

    files = [
        os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".pt")
    ]

    if not files:
        raise ValueError(f"No .pt files found in directory {path}")

    steps = []
    losses = []

    for ckpt in files:
        loss = compute_validation_loss_for_checkpoint(ckpt)
        fname = os.path.basename(ckpt)
        step = extract_step_from_name(fname)
        steps.append(step if step is not None else len(steps))
        losses.append(loss)

    if args.plot:
        plt.figure()
        plt.plot(steps, losses, marker="o")
        plt.xlabel("step")
        plt.ylabel("validation CE loss")
        plt.title("Validation loss vs checkpoint")
        plt.grid()
        out_path = os.path.join(path, "val_sweep.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"Saved sweep plot to {out_path}")


if __name__ == "__main__":
    main()
