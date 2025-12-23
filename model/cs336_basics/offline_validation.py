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
    Create an object from a dotted class path in cfg[field]["name"]
    with kwargs in cfg[field]["params"].
    """
    class_path = cfg[field]["name"]
    cls_params = cfg[field]["params"]

    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)

    if addtl_params is not None:
        cls_params["params"] = addtl_params

    cls = getattr(mod, class_name)
    return cls(**cls_params)


def load_model_from_checkpoint(ckpt_path, device):
    """
    Load LM + cfg from a checkpoint, handling _orig_mod./module. prefixes.
    """
    obj = torch.load(ckpt_path, map_location=device)

    if not isinstance(obj, dict) or "config" not in obj:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not contain a 'config' field. "
            "Did you save the training cfg into the checkpoint?"
        )

    cfg = obj["config"]

    # Build model from config
    LM = create_obj(cfg, field="model").to(device)

    # Clean state dict keys
    state = obj.get("model_state", obj)
    new_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        new_state[k] = v

    missing, unexpected = LM.load_state_dict(new_state, strict=False)
    print(
        f"[load {os.path.basename(ckpt_path)}] missing: {missing} | unexpected: {unexpected}"
    )

    LM.eval()
    return LM, cfg


@torch.inference_mode()
def compute_validation_loss_for_checkpoint(ckpt_path):
    """
    Compute average validation CE loss for a single checkpoint.
    Uses the validation config stored inside the checkpoint.
    """
    # Device selection (similar to your training script)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device} for {ckpt_path}")
    LM, cfg = load_model_from_checkpoint(ckpt_path, device)

    # AMP / dtype logic like training
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

    # Load validation data via memmap (like training)
    val_memmap_data = []
    for d in range(num_shards):
        file_pattern = os.path.join(val_path, f"{val_tag}{d}.bin")
        if not os.path.exists(file_pattern):
            raise FileNotFoundError(f"Validation shard not found: {file_pattern}")
        data_rb = np.memmap(file_pattern, dtype=np.uint16, mode="r")
        val_memmap_data.append(data_rb)

    ce_loss = []
    LM.eval()

    for shard_id in range(num_shards):
        for t in range(num_tries):
            tokenized_data = val_memmap_data[shard_id]
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
        "--pattern",
        default="*.pt",
        help="Glob pattern for checkpoints inside a directory (default: *.pt).",
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

    # Collect checkpoints
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

    # Optional: plot sweep
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
