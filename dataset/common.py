import pickle

import torch
from torchtext.data import Iterator
from tqdm import tqdm


class BucketByLengthIterator(Iterator):
    def __init__(self, *args, max_length=None, example_length_fn=None,
                 data_paths=None, **kwargs):
        batch_size = kwargs['batch_size']

        self.boundaries = self._bucket_boundaries(max_length)
        self.batch_sizes = self._batch_sizes(batch_size)
        self.example_length_fn = example_length_fn
        self.data_paths = data_paths
        self.data_path_idx = 0
        self.buckets = [[] for _ in range(len(self.boundaries)+1)]

        super(BucketByLengthIterator, self).__init__(*args, **kwargs)

    def create_batches(self):
        self.batches = self._bucket_by_seq_length(self.data())

    def reload_examples(self):
        self.data_path_idx = (self.data_path_idx + 1) % len(self.data_paths)
        data_path = self.data_paths[self.data_path_idx]

        examples = torch.load(data_path)
        self.dataset.examples = examples

    def _bucket_by_seq_length(self, data):
        for ex in data:
            length = self.example_length_fn(ex)

            idx = None
            for i, boundary in enumerate(self.boundaries):
                if length <= boundary:
                    idx = i
                    break
            assert idx is not None

            self.buckets[idx].append(ex)
            if len(self.buckets[idx]) >= self.batch_sizes[idx]:
                yield self.buckets[idx]
                self.buckets[idx] = []

    def _bucket_boundaries(self, max_length, min_length=8,
                           length_bucket_step=1.1):
        x = min_length
        boundaries = []
        while x < max_length:
            boundaries.append(x)
            x = max(x + 1, int(x * length_bucket_step))
        return boundaries + [max_length]

    def _batch_sizes(self, batch_size):
        batch_sizes = [
            max(1, batch_size // length) for length in self.boundaries
        ]
        max_batch_size = max(batch_sizes)
        highly_composite_numbers = [
            1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260,
            1680, 2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360,
            50400, 55440, 83160, 110880, 166320, 221760, 277200, 332640,
            498960, 554400, 665280, 720720, 1081080, 1441440, 2162160, 2882880,
            3603600, 4324320, 6486480, 7207200, 8648640, 10810800, 14414400,
            17297280, 21621600, 32432400, 36756720, 43243200, 61261200,
            73513440, 110270160
        ]
        window_size = max(
            [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
        divisors = [i for i in range(1, window_size + 1)
                    if window_size % i == 0]
        return [max([d for d in divisors if d <= bs]) for bs in batch_sizes]


def pickles_to_torch(data_paths):
    print("Refining pickle data...")
    for data_path in tqdm(data_paths, ascii=True):
        examples = []
        with open(data_path, 'rb') as f:
            while True:
                try:
                    example = pickle.load(f)
                except EOFError:
                    break
                examples.append(example)

        with open(data_path, 'wb') as f:
            torch.save(examples, f)
