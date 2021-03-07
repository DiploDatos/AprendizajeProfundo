import csv
import gzip
import random

from torch.utils.data import IterableDataset


class MeliChallengeDataset(IterableDataset):
    def __init__(self,
                 dataset_path,
                 dataset_size,
                 random_buffer_size=2048,
                 transform=None):
        assert random_buffer_size > 0
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size
        self.random_buffer_size = random_buffer_size
        self.transform = transform

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        try:
            with gzip.open(self.dataset_path, "rt") as dataset:
                shuffle_buffer = []
                csv_reader = csv.reader(dataset)
                next(csv_reader)

                for line in csv_reader:
                    _, label_quality, title, category = line

                    if self.random_buffer_size == 1:
                        item = {
                            "title": f"{label_quality} {title}",
                            "category": category
                        }
                        if self.transform:
                            item = self.transform(item)
                        yield item
                    else:
                        shuffle_buffer.append({
                            "title": f"{label_quality} {title}",
                            "category": category
                        })

                        if len(shuffle_buffer) == self.random_buffer_size:
                            random.shuffle(shuffle_buffer)
                            for item in shuffle_buffer:
                                if self.transform:
                                    item = self.transform(item)
                                yield item
                            shuffle_buffer = []

            if len(shuffle_buffer) > 0:
                random.shuffle(shuffle_buffer)
                for item in shuffle_buffer:
                    if self.transform:
                        item = self.transform(item)
                    yield item
        except GeneratorExit:
            return