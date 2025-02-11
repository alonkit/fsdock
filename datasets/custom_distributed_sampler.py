import torch
from torch.utils.data import Sampler, DistributedSampler
import math
from typing import Optional, Iterator

import torch
import torch.distributed as dist

class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.targets, self.indices = dataset.get_partition_and_idxs(num_replicas)
        self.indices_len = list(map(len,self.indices))
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and list(map(len,self.indices)).count(len(self.indices[0])) != len(self.indices):  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = min(self.indices_len)
        else:
            self.num_samples = max(self.indices_len)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.dataset.load(self.targets[rank])

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = [self.indices[self.rank][i] for i in torch.randperm(len(self.indices[self.rank]), generator=g).tolist()]  # type: ignore[arg-type]
        else:
            indices = list(self.indices[self.rank])  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = max(self.indices_len) - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:min(self.indices_len)]

        return iter(indices)
