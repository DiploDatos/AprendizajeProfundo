import csv
import gzip
import logging
import torch

from gensim import corpora
from gensim.parsing import preprocessing
from tqdm import tqdm


DATASET_SIZES = {
    "spanish": {
        "train": 4895281,
        "test": 63681,
        "validation": 1223821
    },
    "portuguese": {
        "train": 5057078,
        "test": 73919,
        "validation": 1264271
    }
}


class RawDataProcessor:
    def __init__(self,
                 dataset_path,
                 dataset_size,
                 ignore_header=True,
                 filters=None,
                 vocab_size=50000):
        if filters:
            self.filters = filters
        else:
            self.filters = [
                lambda s: s.lower(),
                preprocessing.strip_tags,
                preprocessing.strip_punctuation,
                preprocessing.strip_multiple_whitespaces,
                preprocessing.strip_numeric,
                preprocessing.remove_stopwords,
                preprocessing.strip_short,
            ]

        # Create dictionary based on all the reviews (with corresponding preprocessing)
        logging.getLogger().setLevel(logging.ERROR)
        self.dictionary = corpora.Dictionary()
        labels = set()

        with gzip.open(dataset_path, "rt") as dataset:
            csv_reader = csv.reader(dataset)
            if ignore_header:
              next(csv_reader)
            for line in tqdm(csv_reader, total=dataset_size):
                _, label_quality, title, category = line
                self.dictionary.add_documents(
                    [self._preprocess_string(f"{label_quality} {title}")]
                )
                labels.add(category)
        # Filter the dictionary and compactify it (make the indices continous)
        self.dictionary.filter_extremes(no_below=2, no_above=1, keep_n=vocab_size)
        self.dictionary.compactify()
        # Add a couple of special tokens
        self.dictionary.patch_with_special_tokens({
            "[PAD]": 0,
            "[UNK]": 1
        })
        self.idx_to_target = sorted(labels)
        self.target_to_idx = {t: i for i, t in enumerate(self.idx_to_target, start=1)}
        logging.getLogger().setLevel(logging.INFO)

    def _preprocess_string(self, string):
        return preprocessing.preprocess_string(string, filters=self.filters)

    def _sentence_to_indices(self, sentence):
        return self.dictionary.doc2idx(sentence, unknown_word_index=1)

    def encode_data(self, data):
        return self._sentence_to_indices(self._preprocess_string(data))

    def encode_target(self, target):
        return self.target_to_idx.get(target, 0)

    def __call__(self, item):
        if isinstance(item, list):
          data = [self.encode_data(i["title"]) for i in item]
          target = [self.encode_target(i["category"]) for i in item]
        else:
          data = self.encode_data(item["title"])
          target = self.encode_target(item["category"])

        return {
            "data": data,
            "target": target
        }


class PadSequences:
    def __init__(self, pad_value=0, max_length=None, min_length=1):
        assert max_length is None or min_length <= max_length
        self.pad_value = pad_value
        self.max_length = max_length
        self.min_length = min_length

    def __call__(self, items):
        data, target = list(zip(*[(item["data"], item["target"]) for item in items]))
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