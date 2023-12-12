"""utils for ngram for NG_Midiformer model."""

import os
import logging
import math
import numpy as np
import torch
from transformers import cached_path

NGRAM_DICT_NAME = 'ngram.txt'

logger = logging.getLogger(__name__)
class NgramDict(object):

    def __init__(self, ngram_freq_path, tokenizer=None, max_ngram_in_seq=128):
        """Constructs NgramDict

        :param ngram_freq_path: ngrams with frequency
        """
        if os.path.isdir(ngram_freq_path):
            ngram_freq_path = os.path.join(ngram_freq_path, NGRAM_DICT_NAME)
        self.ngram_freq_path = ngram_freq_path
        self.max_ngram_in_seq = max_ngram_in_seq
        self.max_ngram_len = 8
        self.id_to_ngram_list = ["[pad]"]
        self.ngram_to_id_dict = {"[pad]": 0}
        self.ngram_to_freq_dict = {}

        logger.info("loading ngram frequency file {}".format(ngram_freq_path))
        with open(ngram_freq_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                items = line.strip().split(",")
                if len(items) != 2:
                    continue
                ngram, freq = items
                # self.ngram_to_freq_dict[ngram] = int(freq)
                if tokenizer:
                    tokens = tuple(tokenizer.tokenize(ngram))
                    if len([token for token in tokens if "[UNK]" in token]) > 0:
                        tokens = ngram
                else:
                    tokens = tuple(ngram.split(" "))
                self.id_to_ngram_list.append(tokens)
                self.ngram_to_id_dict[tokens] = i + 1
                self.ngram_to_freq_dict[tokens] = int(freq)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, **kwargs):
        ngram_file = pretrained_model_name_or_path
        if os.path.isdir(ngram_file):
            ngram_file = os.path.join(ngram_file, NGRAM_DICT_NAME)
        # redirect to the cache, if necessary
        resolved_ngram_file = cached_path(ngram_file, cache_dir=cache_dir)
        if resolved_ngram_file == ngram_file:
            logger.info("loading vocabulary file {}".format(ngram_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                ngram_file, resolved_ngram_file))
        # Instantiate ngram.
        ngram_dict = cls(resolved_ngram_file, **kwargs)
        return ngram_dict

    def save(self, ngram_freq_path):
        ngram_freq_path = os.path.join(ngram_freq_path, NGRAM_DICT_NAME)
        with open(ngram_freq_path, "w+", encoding="utf-8") as fout:
            for ngram, freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(" ".join(ngram), freq))


def extract_ngram_feature(tokens, ngram_dict, max_seq_len, seg_id_limit):
    # ----------- code for ngram BEGIN-----------
    ngram_matches = []
    #  Filter the word segment from 2 to max_ngram_len to check whether there is a word
    max_gram_n = ngram_dict.max_ngram_len
    tokens=tokens.split(' ')
    for p in range(2, max_gram_n):
        for q in range(0, len(tokens) - p + 1):
            character_segment = tokens[q:q + p]
            # j is the starting position of the word
            # i is the length of the current word
            character_segment = tuple(character_segment)
            if character_segment in ngram_dict.ngram_to_id_dict:
                ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                ngram_freq = ngram_dict.ngram_to_freq_dict[character_segment]
                ngram_matches.append([ngram_index, q, p, character_segment, ngram_freq])

    # shuffle(ngram_matches)
    ngram_matches = sorted(ngram_matches, key=lambda s: s[0])
    # max_word_in_seq_proportion = max_word_in_seq
    max_word_in_seq_proportion = math.ceil((len(tokens) / max_seq_len) * ngram_dict.max_ngram_in_seq)
    if len(ngram_matches) > max_word_in_seq_proportion:
        ngram_matches = ngram_matches[:max_word_in_seq_proportion]
    ngram_ids = [ngram[0] for ngram in ngram_matches]
    ngram_positions = [ngram[1] for ngram in ngram_matches]
    ngram_lengths = [ngram[2] for ngram in ngram_matches]
    ngram_tuples = [ngram[3] for ngram in ngram_matches]
    ngram_freqs = [ngram[4] for ngram in ngram_matches]
    ngram_seg_ids = [0 if position < seg_id_limit else 1 for position in
                     ngram_positions]

    ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
    ngram_mask_array[:len(ngram_ids)] = 1

    # Zero-pad up to the max word in seq length.
    padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
    ngram_ids += padding
    ngram_positions += padding
    ngram_lengths += padding
    ngram_seg_ids += padding
    ngram_freqs += padding

    # ----------- code for ngram END-----------

    return {
        "ngram_ids": ngram_ids,
        "ngram_positions": ngram_positions,
        "ngram_lengths": ngram_lengths,
        "ngram_tuples": ngram_tuples,
        "ngram_seg_ids": ngram_seg_ids,
        "ngram_masks": ngram_mask_array,
        "ngram_freqs": ngram_freqs,
    }


def construct_ngram_matrix(ngram_data, max_seq_length):
    max_ngram_in_sequence = len(ngram_data["ngram_ids"])
    ngram_ids_num = len([x for x in ngram_data["ngram_masks"] if x == 1])

    ngram_positions_matrix = np.zeros(shape=(max_seq_length, max_ngram_in_sequence), dtype=np.float)
    for i in range(ngram_ids_num):
        ngram_positions_matrix[ngram_data["ngram_positions"][i]:
                               ngram_data["ngram_positions"][i] + ngram_data["ngram_lengths"][i], i] = \
            ngram_data["ngram_freqs"][i]
    ngram_positions_matrix_t = torch.from_numpy(ngram_positions_matrix.astype(np.float))
    ngram_positions_matrix_t = torch.div(ngram_positions_matrix_t,
                                         torch.stack([torch.sum(ngram_positions_matrix_t, 1)] * ngram_positions_matrix_t.size(1)).t() + 1e-10)

    return ngram_positions_matrix_t.numpy()
