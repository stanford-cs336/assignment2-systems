import tomllib
import importlib
import numpy as np
import os
import torch
import matplotlib
import argparse
import time

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


import cs336_basics


def benchmark_model(config, benchmark_mode, benchmark_warmup_iter, expt_name):
    """
    training_loop function implements a DL model training loop

    Args :
        config - language model config file
        expt_name - Name of the experiment
    """

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
        amp_flag = False
    else:
        device = torch.device("cpu")
        autocast_dtype = torch.float32
        amp_flag = False

    # Move LM to device
    LM = LM.to(device)

    # Compile the LM
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            LM = torch.compile(LM, mode="max-autotune")
        except Exception as e:
            print(f"torch.compile disabled on CUDA ({e}); continuing uncompiled.")

    # Create the optimizer
    O = create_obj(cfg, field="optimizer", addtl_params=[p for p in LM.parameters()])

    # Run the training loop
    current_epoch = 0
    # learning rate counter
    running_counter = 0
    # bench mark flag
    time_flag = False
    total_time = 0
    time_list = []
    # training flag for pytorch so that grad is available for sure
    LM.train()

    # Training loop
    while current_epoch < total_epochs:
        for _ in range(total_steps_per_epoch):
            tokenized_data = np.memmap("shard_0.bin", dtype=np.uint16, mode="r")
            x, target = cs336_basics.data_loader.data_loader(
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

            # get the learning rate
            current_lr = cs336_basics.learning_rate_schedule.learning_rate_schedule(
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

            ## Sync the CPU GPU threads
            # Generally CPU-GPU executions are async, Meaning, CPU will just go ahead,
            # queuing the instructions in GPU for execution. In order to evaluate,
            # we need the GPU instruction queue empty. So, all the prev. queued instructions
            # need to be executed. So, .synchronize is a blocking call that stops the CPU
            # execution, executes all GPU calls before.
            if running_counter > benchmark_warmup_iter:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()

            # Start timer if running counter > warmup iter
            if running_counter > benchmark_warmup_iter:
                t1 = time.time()
                time_flag = True
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
                loss = cs336_basics.cross_entropy_loss.cross_entropy_loss(y, target)
            # log end timer if only forward is being profiled.
            if benchmark_mode == "forward" and time_flag:
                t2 = time.time()
                total_time = t2 - t1
                time_list.append(total_time)
            # Compute gradients
            loss.backward()

            # Run gradient clipping
            cs336_basics.gradient_clip.gradient_clipping(
                list(LM.parameters()), max_l2_norm=1
            )

            # Run optimizer
            O.step()

            # stop if both forward and backward
            if benchmark_mode == "forward_backward" and time_flag:
                t2 = time.time()
                total_time = t2 - t1
                time_list.append(total_time)
            # update running counter
            running_counter += 1

        # update epoch counter
        current_epoch += 1

    log.info(f"Result : Total Time is {sum(time_list)/len(time_list)} for {expt_name}")


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
    p.add_argument("config_folder", help="Path to config folder")
    p.add_argument("benchmark_mode", help="benchmark either forward/backward/both")
    p.add_argument(
        "warmup_iters", help="number of warm up iters before benchmarking", type=int
    )
    p.add_argument("expt_name", help="Name of the experiment")
    return p.parse_args()


if __name__ == "__main__":
    import logging_setup

    # Example command :
    # python -u cs336_basics/training_loop.py lm_cnfig.toml dummy_run
    arg = get_args()
    breakpoint()
    all_cfg_files = os.listdir(arg.config_folder)
    log, logfile = logging_setup.setup_logger(run_name=arg.expt_name)
    for cfg in all_cfg_files:
        cfg_file = arg.config_folder + "/" + cfg
        log.info(f"Config : {cfg}")
        benchmark_model(cfg_file, arg.benchmark_mode, arg.warmup_iters, arg.expt_name)
