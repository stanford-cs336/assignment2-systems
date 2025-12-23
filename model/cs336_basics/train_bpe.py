import re
import math
import copy
import regex
import multiprocessing as mp
from pretokenization_example import find_chunk_boundaries
from functools import partial
from functools import lru_cache
import pickle
import random
from itertools import islice, pairwise
from collections import Counter, defaultdict


def chunk_dict(data, size):
    """
    Returns a chunk of dictionary
        Args:
        data - input dictionary
        size - size of each dictionary
        Return:
        chunk - smaller dictionary of len size.
    """
    it = iter(data.items())
    while True:
        chunk = dict(islice(it, size))
        if not chunk:
            break
        yield chunk


def give_chunk(boundary_gen, f):
    """
    Yields the chunk from the file for pre tokenizing

    Args:
        boundary_gen - boundary positions of EOS
        f - file id

    Return
        chunk - file chunk
    """
    for start_f, end_f in boundary_gen:
        f.seek(start_f)
        chunk = f.read(end_f - start_f).decode("utf-8", errors="ignore")
        yield chunk


@lru_cache(maxsize=None)
def flatten(sym):
    """
    lru cache remembers some prev. done computation on same inputs.
    Its like if i want to do 1+1 again and again, i can store the out
    of that in cache which is denoted using lru_cache decorator.
    GPT 2 code
    int or nested tuple[int,...] -> flat tuple[int,...]
    """
    if isinstance(sym, int):
        return (sym,)
    out = []
    for x in sym:
        out.extend(flatten(x))
    return tuple(out)


def tuple_to_bytes(t):
    """
    Convert an int or nested tuple of ints into a bytes object.
    Converts the (int,int) to bytes.
    GPT 2 code
    """
    if isinstance(t, int):
        return bytes([t])
    elif isinstance(t, tuple):
        return b"".join(tuple_to_bytes(x) for x in t)
    else:
        raise TypeError(f"Unexpected type in tuple: {type(t)}")


def convert_merges(merge_list):
    return [(tuple_to_bytes(a), tuple_to_bytes(b)) for (a, b) in merge_list]


def pre_tokenize(chunk, special_tkn):
    """
    Pretoken function : pre-tokenizes the corpus
    @in[0]  : chunk         - Input text chunk of type str
    @in[1]  : special_tkn   - Special token list for filter purpose of type list[str]
    @op[0]  : pre_token_dict- output dict of type dict[tuple[bytes], int]
    """

    pre_token_dict = {}
    pre_token_type = "GPT-2"
    # Joins all special tokens.
    # Tricky part. Need to use re.escape
    # Start GPT Code
    join_spl_tkn = "|".join(re.escape(t) for t in special_tkn)
    # End GPT code
    split_corpus = re.split(join_spl_tkn, chunk)
    """
    join_spl_tkn = "|".join(special_tkn)
    # Use re.split across special token. here EoS
    split_corpus = re.split(join_spl_tkn,chunk)
    """
    for txt in split_corpus:
        if txt == "|":
            continue
        if pre_token_type == "white_space":
            # white space split
            pre_token = chunk.split(" ")
            for ch in pre_token:
                if not (tuple(ch) in pre_token_dict.keys()):
                    pre_token_dict[tuple(ch)] = 1
                else:
                    pre_token_dict[tuple(ch)] += 1
        else:
            # Use gpt2 pre tokenizer
            PAT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            pre_token = regex.finditer(PAT, txt)
            for matches in pre_token:
                # here this is in str. so convert to byte
                token_ = matches.group(0).encode("utf-8")
                pre_token_dict[token_] = pre_token_dict.get(token_, 0) + 1
    return pre_token_dict


def train_bpe_fn(file_path, vocab_size, special_tkn_ip):
    """
    train_bpe tokenizes the corpus.
    @in[0] :    file path to the training corpus [str]
    @in[1] :    final size of the vocab [int]
    @in[2] :    list of special tokens [list(str)]
    @op[0] :    final vocab [dict[int,bytes]]
    @op[1] :    bpe merges [list[typles[bytes,bytes]]]
    """
    # Set random seed
    random.seed(42)
    print("\n in train bpe function")
    num_processes = 8
    coarse_chunk_count = 20
    fine_chunk_count = 20
    V = dict()
    # Add the special characters to vocab.
    for i in range(len(special_tkn_ip)):
        V[len(V.keys())] = special_tkn_ip[i].encode("utf-8")

    # Initialize vocabulary
    V.update({i + len(special_tkn_ip): bytes([i]) for i in range(256)})

    # Chunk the corpus and pre-tokenize
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, coarse_chunk_count, b"<|endoftext|>")

        # use sample indices to generate boundaries. sample indices will be [1,4,8,10 etc]
        # choose WHICH chunks to read
        sample_indices = sorted(
            random.sample(range(len(boundaries) - 1), fine_chunk_count)
        )

        # (start, end) for ONLY the sampled chunks
        boundary_gen = [(boundaries[i], boundaries[i + 1]) for i in sample_indices]

        # create a single argument partial function so that we can use map
        worker_fn = partial(pre_tokenize, special_tkn=special_tkn_ip)

        # iterate through result and merge the common entries
        pre_token_dict = {}
        with mp.Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(
                worker_fn, give_chunk(boundary_gen, f), chunksize=4
            ):
                # Increment value
                for k, v in result.items():
                    pre_token_dict[k] = pre_token_dict.get(k, 0) + v

        print("\n pre tokenizaton done")
        merge_list = []
        print("\n create a adjacency pair dict")
        # contains adjacent pair dicts
        adjacent_pair_dict = Counter()
        # contains what keys are containing these pairs in pre token dict
        pairs_keys_dict = defaultdict(set)
        seq_id = {}
        hash_ = 0
        for seq, freq in pre_token_dict.items():
            seq_id[hash_] = seq
            for pairs in pairwise(seq):
                # how many times each pair occur
                # Adjacent pair dict =[(a,b):10,(c,d):20,... ]
                adjacent_pair_dict[pairs] = adjacent_pair_dict.get(pairs, 0) + freq
            for pairs in set(pairwise(seq)):
                pairs_keys_dict[pairs].add(hash_)
            hash_ += 1

        # Global merge happens here
        while len(V.keys()) < vocab_size:
            # Find max key
            max_key = max(
                adjacent_pair_dict,
                key=lambda p: (adjacent_pair_dict[p], flatten(p[0]), flatten(p[1])),
            )
            # find the sequences which max_key appears. list of seq id's.
            # Loop through all keys where the merge has to be done in pre_token_dict
            change_seq = pairs_keys_dict.pop(max_key, set())
            # remove max key from pairs keys dict . this tuple is not valid anymore.
            did_merge = False

            if not change_seq:
                adjacent_pair_dict.pop(max_key)
                continue
            for p in list(change_seq):
                seq_to_proc = seq_id[p]
                # get this freq
                pre_token_freq = pre_token_dict[seq_to_proc]
                # remove this old seq
                pre_token_dict.pop(seq_to_proc)
                list_form = list(seq_to_proc)
                mod_list = []
                itr = 0
                while itr < len(list_form) - 1:
                    if (
                        list_form[itr] == max_key[0]
                        and list_form[itr + 1] == max_key[1]
                    ):
                        mod_list.append((max_key))
                        itr += 2
                        # need this flag to prevent adding phantom merges.
                        did_merge = True
                    else:
                        mod_list.append(list_form[itr])
                        itr += 1
                if itr == len(list_form) - 1:  # handle leftover
                    mod_list.append(list_form[itr])
                merged = tuple(mod_list)
                # add the merged key. The freq is same as 'p'
                pre_token_dict[merged] = pre_token_dict.get(merged, 0) + pre_token_freq
                # update seq id
                seq_id[p] = merged
                # Update adjacent pair dicts
                old_pairs = Counter(pairwise(seq_to_proc))
                new_pairs = Counter(pairwise(merged))
                # compute difference
                combined_pairs = old_pairs.keys() | new_pairs.keys()
                difference_ = Counter()
                for c in combined_pairs:
                    diff_pair = new_pairs.get(c, 0) - old_pairs.get(c, 0)
                    # process only non zeros
                    if diff_pair:
                        difference_[c] = pre_token_freq * (diff_pair)

                adjacent_pair_dict.update(difference_)
                # Update the pairs - keys dict
                # Add new pair in pairs-keys dict
                for oldies in set(old_pairs):
                    oldie_seq_vals = pairs_keys_dict.get(oldies)
                    if oldie_seq_vals:
                        oldie_seq_vals.discard(p)
                        if not oldie_seq_vals:
                            # remove this item from this key
                            pairs_keys_dict.pop(oldies, None)

                # create new pairs-keys dict with merged tokens.
                for newbie in set(new_pairs):
                    # This is new key. Add this to the pairs_keys_dict.
                    pairs_keys_dict[newbie].add(p)

            # remove 0 entries.
            for k, _ in list(adjacent_pair_dict.items()):
                if adjacent_pair_dict[k] <= 0:
                    adjacent_pair_dict.pop(k)

            # Init freq dict every time I run merge
            max_key_bytes = tuple_to_bytes(max_key)

            # update pair counts
            # remove current item from the pair
            if did_merge:
                merge_list.append(max_key)
                if not (max_key_bytes in V.values()):
                    V[len(V)] = max_key_bytes
            else:
                adjacent_pair_dict.pop(max_key)

            if len(V) % 5 == 0:
                print("\n length of vocab - ", len(V))

    merge_list = convert_merges(merge_list)
    # Save to JSON file
    with open("tiny_stories_train_vocab.pkl", "wb") as f:
        pickle.dump(V, f)

    with open("tiny_stories_train_merges.pkl", "wb") as f:
        pickle.dump(merge_list, f)
    return V, merge_list


if __name__ == "__main__":
    file_name = "data/TinyStoriesV2-GPT4-valid.txt"
    num_cores = mp.cpu_count()
    print(f"Number of logical CPU cores: {num_cores}")
    train_bpe_fn(file_name, 10000, ["<|endoftext|>"])
