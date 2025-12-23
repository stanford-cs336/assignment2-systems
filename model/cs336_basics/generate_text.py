import torch
import tomllib
import importlib
import pickle
import regex
import math
import re
import numpy as np


def generate_text(x, len_x, temp, top_p):
    """
    generate_text uses the trained model and generates text.

    Args :
        x - Input prompt (string)
        len_x - Maximum length of output expected (int)
        temp - softmax temperature

    Return :
        x - Completion to x (string)
    """
    # create the model object
    with open(
        "/Users/hari/Documents/backups/Expts/run_pod_3/lm_config.toml", "rb"
    ) as f:
        cfg = tomllib.load(f)

    LM = create_obj(cfg, field="model")
    device = LM.device
    LM = LM.to(device)

    # load the language model checkpoint (handles dict payloads)
    LM_state = torch.load(
        "/Users/hari/Documents/backups/Expts/run_pod_3/checkpoint/run_pod_3.pt",
        map_location=device,
    )
    state_dict = LM_state.get("model_state", LM_state)
    cleaned_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("module."):
            k = k[len("module.") :]
        cleaned_state[k] = v
    LM.load_state_dict(cleaned_state, strict=True)

    # Tokenize the input string
    with open("tiny_stories_train_vocab.pkl", "rb") as f:
        V = pickle.load(f)
    with open("tiny_stories_train_merges.pkl", "rb") as file:
        M = pickle.load(file)

    # Find which index in V is end of text
    for v in range(len(V)):
        if V[v] == b"<|endoftext|>":
            eot_id = v
            break

    special_tokens = ("<|endoftext|>",)

    # inference loop
    x_token = encode(x, specials=special_tokens, V=V, merge_lst=M)
    # x_token is in np.array. convert to torch. Add a 1D leading dim
    # as it expects in BxT
    x_token = torch.from_numpy(x_token).unsqueeze(0).to(device=device, dtype=torch.long)
    new_tokens = []
    max_seq_len = cfg["model"]["params"]["context_length"]
    rope_theta = cfg["training"]["params"]["theta"]

    if temp <= 0:
        raise ValueError("Temperature must be > 0 for sampling.")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1].")

    with torch.inference_mode():
        while len(new_tokens) < len_x:
            # Trim context window before passing to the model
            model_input = x_token[:, -max_seq_len:]
            token_positions = torch.arange(
                model_input.size(1), device=device, dtype=torch.long
            )

            # Get the logits
            y = LM.tranform_lm_model(
                model_input,
                rope_theta=rope_theta,
                token_positions=token_positions,
                max_seq_len=max_seq_len,
            )

            # Get the softmax output
            p = softmax_with_temp(y[:, -1, :], -1, temp).squeeze(0)

            # top p sampling
            sorted_p, sort_idx = p.sort(descending=True)
            cum_p = torch.cumsum(sorted_p, dim=0)
            cutoff_idx = torch.where(cum_p >= top_p)[0]
            top_count = cutoff_idx[0].item() + 1 if len(cutoff_idx) else len(sorted_p)
            top_p_elem = sort_idx[:top_count]
            top_p_prob = sorted_p[:top_count]

            # new prob
            top_p_mod = top_p_prob / top_p_prob.sum()
            # select the index to pick from the vocab
            chosen_index = torch.multinomial(top_p_mod, num_samples=1)
            token_chosen = top_p_elem[chosen_index]
            # find the byte in V
            next_token = V[token_chosen.item()]
            # Append this into the list of new tokens generated
            new_tokens.append(next_token)
            # convert the byte to string
            token_to_str = next_token.decode()
            # append the new string to the old string
            x = x + token_to_str
            # add the token to the x_token ( id form to id form )
            token_tensor = token_chosen.view(1, 1).to(device)
            x_token = torch.cat((x_token, token_tensor), dim=1)

            # break if this is EOT
            if token_chosen.item() == eot_id:
                break

    return x


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


def softmax_with_temp(x, dim, temp):
    x_d_norm = x - torch.max(x, dim=dim, keepdim=True).values
    x_d_sum = torch.sum(torch.exp(x_d_norm / temp), dim=dim, keepdim=True)
    y = torch.exp(x_d_norm / temp) / x_d_sum
    return y


def encode(text, specials, V, merge_lst):
    final_token = []
    # reverse merge dict
    merge_rank = {merge_lst[i]: i for i in range(len(merge_lst))}

    if specials:
        # If multiple specials, sort by length desc so longer matches win first
        specials_sorted = sorted(specials, key=len, reverse=True)
        special_pat = "(" + "|".join(map(re.escape, specials_sorted)) + ")"
        # Use the same engine as your PAT (you used `regex`), so:
        split_text = regex.split(
            special_pat, text  # type: ignore
        )  # pyright: ignore[reportArgumentType, reportCallIssue]
    else:
        split_text = [text]
    # Map tokens to IDs
    # reverse dict construction
    reverse_dict = {value: key for key, value in V.items()}
    for s_t in split_text:
        if specials and s_t in specials:
            s_t_bytes = s_t.encode(  # type: ignore
                "utf-8"
            )  # pyright: ignore[reportOptionalMemberAccess]
            final_token.append(reverse_dict[s_t_bytes])
        elif s_t == "":
            continue
        else:
            PAT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            pre_token = regex.finditer(PAT, s_t)  # type: ignore
            # read merge list and create a reverse dict
            for matches in pre_token:
                # ex : "cat is here" is the string
                # temp str is in bytes.[b'c',b'a',b't']
                # have to find the non overlapping combos and merge based on priority
                bs = matches.group(0).encode("utf-8")
                temp_str = [bytes([b]) for b in bs]
                # combos are [(c,a),(a,t)]
                combos = list(zip(temp_str[:-1], temp_str[1:]))
                # find in self.merge_rank to find out if it is present
                combo_rank = []
                for c in range(len(combos)):
                    temp_rank = merge_rank.get(combos[c])
                    combo_rank.append(
                        (temp_rank, c) if temp_rank is not None else (math.inf, c)
                    )
                while True:
                    if any(c[0] < math.inf for c in combo_rank):
                        # atleast one merge is present. picks lexi. smalled tuple
                        min_find = min(combo_rank)
                        # merge the min pair
                        temp_str[min_find[1]] = (
                            temp_str[min_find[1]] + temp_str[min_find[1] + 1]
                        )
                        # modify combos to get token
                        temp_str[min_find[1] + 1 :] = temp_str[min_find[1] + 2 :]
                        # bump up the old rank so that its irrelevant.
                        combo_rank.pop(min_find[1])
                        # reduce the idx on right to end of combo_rank by 1
                        for idx in range(min_find[1], len(combo_rank)):
                            offset_idx = idx
                            combo_rank[offset_idx] = (
                                combo_rank[offset_idx][0],
                                combo_rank[offset_idx][1] - 1,
                            )
                        if min_find[1] - 1 >= 0:
                            # recompute the new rank for lower idx
                            lower_rank = merge_rank.get(
                                (
                                    temp_str[min_find[1] - 1],
                                    temp_str[min_find[1]],
                                ),
                                math.inf,
                            )
                            combo_rank[min_find[1] - 1] = (
                                lower_rank,
                                min_find[1] - 1,
                            )
                        if min_find[1] < len(combo_rank):
                            # recompute the new rank for higher idx
                            higher_rank = merge_rank.get(
                                (
                                    temp_str[min_find[1]],
                                    temp_str[min_find[1] + 1],
                                ),
                                math.inf,
                            )
                            combo_rank[min_find[1]] = (
                                higher_rank,
                                min_find[1],
                            )
                    else:
                        for v in temp_str:
                            final_token.append(reverse_dict[v])
                        break
    as_array = np.asarray(final_token, dtype=np.uint16)
    return as_array


if __name__ == "__main__":
    from cs336_basics.generate_text import generate_text

    prompt = "Once there lived a dog called bruno"
    completion = generate_text(prompt, len_x=256, temp=0.4, top_p=0.75)
    print(completion)
