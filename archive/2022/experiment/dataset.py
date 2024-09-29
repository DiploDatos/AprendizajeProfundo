import gzip
import json
import random

from torch.utils.data import IterableDataset


class MeliChallengeDataset(IterableDataset):
    def __init__(self,
                 dataset_path,
                 random_buffer_size=2048):
        assert random_buffer_size > 0
        self.dataset_path = dataset_path
        self.random_buffer_size = random_buffer_size

        with gzip.open(self.dataset_path, "rt") as dataset:
            item = json.loads(next(dataset).strip())
            self.n_labels = item["n_labels"]
            self.dataset_size = item["size"]

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        try:
            with gzip.open(self.dataset_path, "rt") as dataset:
                shuffle_buffer = []

                for line in dataset:
                    item = json.loads(line.strip())
                    item = {
                        "data": item["data"],
                        "target": item["target"]
                    }

                    if self.random_buffer_size == 1:
                        yield item
                    else:
                        shuffle_buffer.append(item)

                        if len(shuffle_buffer) == self.random_buffer_size:
                            random.shuffle(shuffle_buffer)
                            for item in shuffle_buffer:
                                yield item
                            shuffle_buffer = []

                if len(shuffle_buffer) > 0:
                    random.shuffle(shuffle_buffer)
                    for item in shuffle_buffer:
                        yield item
        except GeneratorExit:
            return
