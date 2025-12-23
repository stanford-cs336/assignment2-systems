import regex
import re
import multiprocessing as mp
import pickle
import numpy as np
import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
from functools import partial
import math


# from train_bpe import train_bpe_fn
class Tokenizer:
    _worker_singleton = None

    def __init__(
        self, vocab, merges, special_token=None, file_name=None, output_fol=None
    ):
        self.V = vocab
        self.merge_lst = merges
        self.spl_tkn = special_token or []
        # create a merge rank
        self.merge_rank = {self.merge_lst[i]: i for i in range(len(self.merge_lst))}
        print(file_name)
        self.file_name = file_name
        self.output_loc = output_fol
        self.test_mode = False

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        file_in: str,
        file_out: str,
        special_token=None,
    ):
        # Load Vocab and merges
        with open(vocab_filepath, "rb") as file:
            V = pickle.load(file)
        with open(merges_filepath, "rb") as file:
            M = pickle.load(file)
        return cls(V, M, special_token, file_in, file_out)

    _worker_singleton = None  # one instance per worker

    @classmethod
    def init_worker(
        cls, vocab_path, merges_path, specials, file_in: str, file_out: str
    ):
        """Runs once in each worker process."""
        cls._worker_singleton = cls.from_files(
            vocab_path, merges_path, file_in, file_out, specials
        )

    @classmethod
    def encode_worker(cls, boundary_tup):
        """pair = (idx, text)  ->  (idx, token_ids)"""
        return cls._worker_singleton.encode(boundary_tup=boundary_tup)  # type: ignore

    def encode(self, boundary_tup):

        final_token = []
        specials = self.spl_tkn  # extend if you have more
        if not self.test_mode:
            if boundary_tup is not None:
                # read the file
                if self.file_name != None:
                    with open(self.file_name, "rb") as f:
                        shard_idx, start_f, end_f = boundary_tup
                        f.seek(start_f)
                        text = f.read(end_f - start_f).decode("utf-8", errors="ignore")
            else:
                assert 0, "No boundary tuples"
        else:
            text = boundary_tup

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
        reverse_dict = {value: key for key, value in self.V.items()}
        for s_t in split_text:
            if s_t in specials:
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
                        temp_rank = self.merge_rank.get(combos[c])
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
                                lower_rank = self.merge_rank.get(
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
                                higher_rank = self.merge_rank.get(
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
        if not self.test_mode:
            as_array = np.asarray(final_token, dtype=np.uint16)
            bin_file = self.output_loc + "/shard_" + str(shard_idx) + ".bin"  # type: ignore
            as_array.tofile(bin_file)
            print("saving file ", shard_idx)
            return shard_idx
        else:
            return final_token

    def encode_iterable(self, iterable):
        for m in iterable:
            for e in self.encode(m):
                yield e

    def decode(self, ids: list[int]) -> str:
        tokens = b""
        for v in ids:
            tokens = tokens + self.V[v]
        token_to_str = tokens.decode()
        return token_to_str


if __name__ == "__main__":
    file_name = "data/TinyStoriesV2-GPT4-valid.txt"
    file_size = os.path.getsize(file_name)
    total_batches = 10
    output_folder = "tiny_stories_valid_token_out"

    # create a folder for token output. # change in line 203 also.
    os.makedirs(output_folder, mode=0o755, exist_ok=True)

    with open(file_name, "rb") as f:
        boundaries = find_chunk_boundaries(f, total_batches, b"<|endoftext|>")
        # (start, end) for ONLY the sampled chunks
        boundary_gen = [
            (i, boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
        ]
    # create a single argument partial function so that we can use map
    print("Going to process")
    with mp.Pool(
        processes=1,
        initializer=Tokenizer.init_worker,
        initargs=(
            "tiny_stories_train_vocab.pkl",
            "tiny_stories_train_merges.pkl",
            ["<|endoftext|>"],
            file_name,
            output_folder,
        ),
    ) as pool:
        for shard in pool.imap_unordered(
            Tokenizer.encode_worker, boundary_gen, chunksize=64
        ):
            print("done : ", shard)
