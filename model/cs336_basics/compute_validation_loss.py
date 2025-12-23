import tomllib
import importlib
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import argparse
import pickle
import torch.nn.functional as F

from cs336_basics import data_loader
from cs336_basics import cross_entropy_loss
from cs336_basics import checkpoint_utils


@torch.inference_mode()
def compute_validation_loss(checkpoint, config):
    """
    training_loop function implements a DL model training loop

    Args :
        config - language model config file
    """

    log.info("Starting validation")
    # read the cfg
    with open(config, "rb") as f:
        cfg = tomllib.load(f)

    # Num of shards
    num_shards = cfg["validation"]["data"]["num_shards"]
    # Pick 25 different items per shard
    num_tries = cfg["validation"]["params"]["num_tries"]
    # Loss array
    ce_loss = []
    # model  & training params
    B = cfg["training"]["params"]["batch_size"]
    T = cfg["model"]["params"]["context_length"]
    theta = cfg["training"]["params"]["theta"]

    # Define model params.
    LM = create_obj(cfg, field="model")

    # set the device properly
    if LM.device.type == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = (
            True  # TF32 for any remaining FP32 matmuls
        )
        torch.backends.cudnn.allow_tf32 = True
        # autocast data type for optimization
        autocast_dtype = torch.bfloat16
        amp_flag = True
    elif LM.device.type == "mps":
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        # autocast data type for optimization
        autocast_dtype = torch.float16
        amp_flag = True
    else:
        device = torch.device("cpu")
        autocast_dtype = torch.float32
        amp_flag = False

    # Move LM to device
    LM = LM.to(device)
    # === LOAD CHECKPOINT *BEFORE* COMPILE ===
    _ckpt = load_any_checkpoint(checkpoint, LM, device)

    # --- Zero-logits sanity (should equal ln(V)) ---
    B = cfg["training"]["params"]["batch_size"]
    T = cfg["model"]["params"]["context_length"]

    LM.eval()
    LM.float()
    amp_flag = False
    autocast_dtype = torch.float32

    # load the final_model
    # load_any_checkpoint(checkpoint, LM, device)

    # Compile
    if hasattr(torch, "compile") and device.type == "cuda":
        LM = torch.compile(LM, mode="max-autotune")

    # load the data from .bin
    file_ids = range(num_shards)
    len_x = []
    for l in file_ids:
        file_pattern = "tiny_stories_valid_token_out/shard_" + str(l) + ".bin"
        size_per_file = os.path.getsize(file_pattern)
        unit_memory = 2  # bytes
        num_tokens_per_file = size_per_file / unit_memory
        len_x.append(num_tokens_per_file)

    # Read once all the data
    memmap_data = []

    # Eval mode
    # LM.eval()
    for d in range(num_shards):
        file_pattern = "tiny_stories_valid_token_out/shard_" + str(d) + ".bin"
        data_rb = np.memmap(file_pattern, dtype=np.uint16, mode="r")
        memmap_data.append(data_rb)

    # validation loop
    # generate the data for this epoch
    for b_id in range(num_shards):
        for b in range(num_tries):
            # load this shard
            tokenized_data = memmap_data[b_id]
            x, target = data_loader.data_loader(
                tokenized_data,
                batch_size=B,
                context_length=T,
                device_type=device.type,
            )

            # safe to move again
            x = x.to(device=device)
            target = target.to(device=device)

            # token positions
            token_pos = torch.arange(T, device=device)

            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=amp_flag
            ):
                # Run the language model
                logits = LM.tranform_lm_model(
                    x,
                    rope_theta=theta,
                    token_positions=token_pos,
                    max_seq_len=x.size(1),
                )

                # compute entropy loss
                loss = cross_entropy_loss.cross_entropy_loss(logits, target)
            # Status update
            # add the loss in ce_loss for plot
            ce_loss.append(loss.item())
            log.info(f"shard={b_id} try={b} loss={loss.item():.4f}")

    print(f"Final loss: {sum(ce_loss) / len(ce_loss):.4f}")
    return sum(ce_loss) / len(ce_loss)


def create_obj(cfg, field, addtl_params=None):
    class_path = cfg[field]["name"]
    cls_params = cfg[field]["params"]

    # Split path into module and class
    module_name, class_name = class_path.rsplit(".", 1)

    # Dynamically import module
    mod = importlib.import_module(module_name)

    # Add config params
    if addtl_params != None:
        # this is mainly for handling optimizer
        cls_params["params"] = addtl_params

    # Retrieve class object from module
    cls = getattr(mod, class_name)
    obj = cls(**cls_params)
    return obj


def get_args():
    p = argparse.ArgumentParser(description="Short description of your script.")
    # positional
    p.add_argument("config_name", help="Path to config file")
    p.add_argument("expt_path", help="expt path containing .pt files")
    p.add_argument("validation_log_name", help="val run log name")
    return p.parse_args()


def load_any_checkpoint(path, LM, device):
    obj = torch.load(path, map_location=device)
    breakpoint()
    state = obj.get("model_state", obj) if isinstance(obj, dict) else obj
    # Fix keys from compiled model (_orig_mod.) or DataParallel (module.)
    new_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        new_state[k] = v

    state = new_state
    missing, unexpected = LM.load_state_dict(state, strict=False)
    print(f"load_state_dict missing: {missing} unexpected: {unexpected}")
    return obj


if __name__ == "__main__":
    from cs336_basics import logging_setup
    import pickle

    arg = get_args()

    log, logfile = logging_setup.setup_logger(run_name=arg.validation_log_name)

    extension = [".pt"]
    loss_lst = []
    for fn in os.listdir(arg.expt_path):
        full = os.path.join(arg.expt_path, fn)
        if os.path.isfile(full) and os.path.splitext(fn)[1].lower() in extension:
            log.info(f"Validation run on {full}")
            ce_loss = compute_validation_loss(full, arg.config_name)
            loss_lst.append(ce_loss)

    # plot the loss
    plt.figure()
    plt.plot(loss_lst, label="loss")
    plt.xlabel("step")
    plt.ylabel("train CE loss")
    plt.title("lr sweep")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("lr_sweep.png", dpi=200)  # png/jpg/pdf/svg all work
