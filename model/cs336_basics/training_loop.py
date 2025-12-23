import tomllib
import importlib
import numpy as np
import os
import torch
import matplotlib
import argparse
import wandb

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import json

from cs336_basics import learning_rate_schedule
from cs336_basics import data_loader
from cs336_basics import cross_entropy_loss
from cs336_basics import gradient_clip


def dump_config(cfg_dict):
    log.info("CONFIG:\n" + json.dumps(cfg_dict, indent=2, sort_keys=True))


def validate(LM, val_data, cfg, device, autocast_dtype, amp_flag):
    """
    Validates the model in the middle of training

    Args :
        LM : Language model
        val_Data : validation shard file paths
        cfg : config file
        device : device in which validation has to happen
    """
    # model params
    B = cfg["training"]["params"]["batch_size"]
    T = cfg["model"]["params"]["context_length"]
    theta = cfg["training"]["params"]["theta"]
    # Validation params
    validation_num_shards = cfg["validation"]["data"]["num_shards"]
    validation_num_tries = cfg["validation"]["params"]["num_tries"]
    # validation loss
    ce_loss = []

    # validation loop
    LM.eval()
    for b_id in range(validation_num_shards):
        shard_file = val_data[b_id]
        tokenized_data = np.memmap(shard_file, dtype=np.uint16, mode="r")
        try:
            for b in range(validation_num_tries):
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

                with torch.no_grad():
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
                ce_loss.append(loss.item())
        finally:
            if hasattr(tokenized_data, "_mmap"):
                tokenized_data._mmap.close()

    final_val_loss = sum(ce_loss) / len(ce_loss)
    return final_val_loss


def training_loop(config, expt_name):
    """
    training_loop function implements a DL model training loop

    Args :
        config - language model config file
        expt_name - Name of the experiment
    """

    log.info("Starting training")

    # Open config
    with open(config, "rb") as f:
        cfg = tomllib.load(f)

    # Variable initializations
    # To tokenize data
    total_epochs = cfg["training"]["params"]["epochs"]
    total_steps_per_epoch = cfg["training"]["params"]["steps_per_epoch"]
    # Num of shards
    folder_path = cfg["data"]["params"]["path"]
    file_tag = cfg["data"]["params"]["tag"]
    num_shards = cfg["data"]["params"]["num_shards"]
    offset_file = 900
    # Loss array
    ce_loss = []
    ema_ce_loss = []
    # checkpoint variables
    checkpoint_every = cfg["training"]["params"]["checkpoint_every"]
    log_every_steps = 20
    # model  & training params
    model_name = cfg["model"]["name"]
    B = cfg["training"]["params"]["batch_size"]
    T = cfg["model"]["params"]["context_length"]
    theta = cfg["training"]["params"]["theta"]
    shards_used = torch.zeros(total_steps_per_epoch, device="cpu", dtype=torch.long)
    # LR params
    alpha_max = cfg["training"]["params"]["alpha_max"]
    alpha_min = cfg["training"]["params"]["alpha_min"]
    warmup_iter = cfg["training"]["params"]["warmup_iter"]
    num_cosine_iter = cfg["training"]["params"]["cooldown_iter"]
    # Validation params
    validation_data_path = cfg["validation"]["data"]["path"]
    validate_every_steps = cfg["validation"]["params"]["validate_every_steps"]
    validation_data_tag = cfg["validation"]["data"]["tag"]
    validation_num_shards = cfg["validation"]["data"]["num_shards"]
    val_loss = []
    val_steps = []
    # dump config file in log
    dump_config(cfg)

    log.info(f"steps per epoch = {total_steps_per_epoch:.3e}")

    # weights & biases for training
    log.info("Initializing wandb")
    run = wandb.init(
        # Set the wandb project where this run will be logged.
        project="llm_train_project",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": alpha_max,
            "architecture": model_name,
            "dataset": folder_path,
            "epochs": total_epochs,
        },
    )

    # Create model object
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

    log.info(f"running the code in {device} ")

    # Move LM to device
    LM = LM.to(device)

    # Compile the LM
    if hasattr(torch, "compile"):
        mode = "max-autotune" if device.type == "cuda" else "reduce-overhead"
        try:
            LM = torch.compile(LM, mode=mode)
        except Exception as e:
            log.warning(f"torch.compile disabled ({e}); continuing uncompiled.")

    # Create the optimizer
    O = create_obj(cfg, field="optimizer", addtl_params=[p for p in LM.parameters()])

    # Training data preparation
    # load the data from .bin & compute the pdf for selection in training
    file_ids = range(num_shards)
    len_x = []
    shard_paths = []
    for l in file_ids:
        file_pattern = os.path.join(folder_path, f"{file_tag}{l}.bin")
        size_per_file = os.path.getsize(file_pattern)
        unit_memory = 2  # bytes
        num_tokens_per_file = size_per_file / unit_memory
        len_x.append(num_tokens_per_file)
        shard_paths.append(file_pattern)

    # compute pdf of training data
    total_tokens = sum(len_x)
    len_weight = np.asarray([(x / total_tokens) for x in len_x])

    # validation data preparation
    # Prepare list of validation shard paths
    val_shard_paths = []
    for d in range(validation_num_shards):
        file_pattern = os.path.join(
            validation_data_path, f"{validation_data_tag}{offset_file + d}.bin"
        )
        val_shard_paths.append(file_pattern)

    # Run the training loop
    current_epoch = 0
    # learning rate counter
    running_counter = 0
    # Training flag
    stop_train = False
    # training flag for pytorch so that grad is available for sure
    LM.train()
    # Training loop
    while current_epoch < total_epochs and not (stop_train):
        # generate the data for this epoch
        shards_used.zero_()
        for b_id in range(total_steps_per_epoch):
            # different shards to memmap every epoch
            choice_shards = np.random.choice(
                np.asarray(file_ids), size=1, p=len_weight
            )[0]
            shards_used[b_id] = int(choice_shards)
            # load this shard
            shard_file = shard_paths[choice_shards]
            tokenized_data = np.memmap(shard_file, dtype=np.uint16, mode="r")
            try:
                x, target = data_loader.data_loader(
                    tokenized_data,
                    batch_size=B,
                    context_length=T,
                    device_type=device.type,
                )
            finally:
                if hasattr(tokenized_data, "_mmap"):
                    tokenized_data._mmap.close()
            # safe to move again
            x = x.to(device=device)
            target = target.to(device=device)

            # token positions
            token_pos = torch.arange(T, device=device)

            # get the learning rate
            current_lr = learning_rate_schedule.learning_rate_schedule(
                running_counter,
                alpha_max=alpha_max,
                alpha_min=alpha_min,
                warmup_iter=warmup_iter,
                num_cosine_iter=num_cosine_iter,
            )

            # update the learning rate of optimizer
            for i in range(len(O.param_groups)):
                O.param_groups[i]["lr"] = current_lr

            # zero the grad before computing grad
            O.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, enabled=amp_flag
            ):
                # Run the language model
                y = LM.tranform_lm_model(
                    x,
                    rope_theta=theta,
                    token_positions=token_pos[: x.size(1)],
                    max_seq_len=x.size(1),
                )
                loss = cross_entropy_loss.cross_entropy_loss(y, target)

            # Compute gradients
            loss.backward()

            # Run gradient clipping
            gradient_clip.gradient_clipping(list(LM.parameters()), max_l2_norm=1)

            # Run optimizer
            O.step()

            # update running counter
            running_counter += 1

            # Checkpointing
            if running_counter % checkpoint_every == 0:
                pkg = {
                    "model_state": LM.state_dict(),
                    "optimizer_state": O.state_dict(),
                    "iter": running_counter,
                    "config": cfg,
                }
                latest_checkpoint = f"artifacts/checkpoint/iter_{running_counter}.pt"
                os.makedirs("artifacts/checkpoint", exist_ok=True)
                torch.save(pkg, latest_checkpoint)

            # Validation run
            if running_counter % validate_every_steps == 0:
                # get the latest checkpoint
                # run validate and append in val_loss
                val_loss.append(
                    validate(
                        LM=LM,
                        val_data=val_shard_paths,
                        cfg=cfg,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        amp_flag=amp_flag,
                    )
                )
                val_steps.append(running_counter)
                # restore LM Mode
                LM.train()

            # Log loss
            if running_counter % log_every_steps == 0:
                # add the loss in ce_loss for plot
                ce_loss.append(loss.item())
                if ema_ce_loss != []:
                    curr_ema = 0.8 * ema_ce_loss[-1] + 0.2 * loss.item()
                    ema_ce_loss.append(curr_ema)
                else:
                    ema_ce_loss.append(loss.item())

                last_val = val_loss[-1] if val_loss else float("nan")

                log.info(
                    f"step={running_counter} loss={loss.item():.4f} lr={current_lr:.3e} val_loss={last_val}"
                )
                # wandb counter
                log_dict_wandb = {
                    "train_loss": loss.item(),
                    "ema_loss": ema_ce_loss[-1],
                    "lr": current_lr,
                }
                if val_loss != []:
                    log_dict_wandb["val_loss"] = last_val

                run.log(
                    log_dict_wandb,
                    step=running_counter,
                )

            if loss.item() < 1.8:
                # break the training as loss < 1.8
                stop_train = True
                break
        # update epoch counter
        current_epoch += 1
        # save the choice of shards used in training to see diversity
        torch.save(
            shards_used,
            f"artifacts/logs/{expt_name}_choice_shards_epoch_{current_epoch}.pt",
        )

    if "pkg" not in locals():
        pkg = {
            "model_state": LM.state_dict(),
            "optimizer_state": O.state_dict(),
            "iter": running_counter,
            "config": cfg,
        }

    # save the model
    torch.save(pkg, f"artifacts/checkpoint/{expt_name}.pt")
    # plot the loss
    steps_train = list(range(len(ce_loss)))
    plt_title = expt_name + "-Training loss Vs Validation loss"

    plt.figure()
    plt.plot(val_steps, val_loss, label="validation loss")
    plt.plot(steps_train, ema_ce_loss, label="EMA train loss")
    plt.xlabel("step")
    plt.ylabel("CE loss")
    plt.title(plt_title)

    plt.legend()
    plt.tight_layout()
    plt.grid()

    plt.savefig("artifacts/logs/loss.png", dpi=600)
    # Also print final values
    print(f"Final loss: {ce_loss[-1]:.4f}")
    log.info("Training complete")
    # complete wandb
    run.log({"loss_curve": wandb.Image("artifacts/logs/loss.png")})
    run.finish()


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
    p.add_argument("expt_name", help="expt name for tag")
    return p.parse_args()


if __name__ == "__main__":
    from cs336_basics import logging_setup

    # Example command :
    # python -u cs336_basics/training_loop.py lm_cnfig.toml dummy_run
    arg = get_args()

    os.makedirs("artifacts/checkpoint", exist_ok=True)
    os.makedirs("artifacts/logs", exist_ok=True)

    log, logfile = logging_setup.setup_logger(run_name=arg.expt_name)
    training_loop(arg.config_name, arg.expt_name)
