import csv
import gzip
import logging
import torch

from gensim import corpora
from gensim.parsing import preprocessing
from tqdm import tqdm


class PadSequences:
    def __init__(self, pad_value=0, max_length=None, min_length=1):
        assert max_length is None or min_length <= max_length
        self.pad_value = pad_value
        self.max_length = max_length
        self.min_length = min_length

    def __call__(self, items):
        data = [item["data"] for item in items]
        target = [item["target"] for item in items]
        seq_lengths = [len(d) for d in data]

        if self.max_length:
            max_length = self.max_length
            seq_lengths = [min(self.max_length, l) for l in seq_lengths]
        else:
            max_length = max(self.min_length, max(seq_lengths))

        data = [d[:l] + [self.pad_value] * (max_length - l)
                for d, l in zip(data, seq_lengths)]

        return {
            "data": torch.LongTensor(data),
            "target": torch.LongTensor(target)
        }
