import random
import numpy as np
import torchdata.datapipes as dp
from torch.utils.data.datapipes.datapipe import IterDataPipe
from freerec.data.datasets import RecDataSet
from freerec.data.tags import ITEM, ID
from freerec.data.postprocessing.base import BaseProcessor

from partition import Mope


class GroupLauncher(dp.iter.IterDataPipe):

    def __init__(
        self, partition: Mope, shuffle=False
    ) -> None:
        super().__init__()

        self.partition = partition
        self.shuffle = shuffle
        self._rng = random.Random()
        self.set_seed(0)

    def set_seed(self, seed: int):
        seed = int(seed) % (2**31 - 1)
        self._rng.seed(seed)

        groups = self.partition(seed)
        self.source = []
        for label in np.unique(groups):
            self.source.append(
                np.flatnonzero(groups == label).tolist()
            )

    def __iter__(self):
        r"""Yield indices, optionally shuffled."""
        if self.shuffle:
            self._rng.shuffle(self.source)
        for items in self.source:
            if self.shuffle:
                self._rng.shuffle(items)
            yield from iter(items)


class GroupSource(BaseProcessor):

    def __init__(
        self,
        dataset: RecDataSet, partition: Mope,
        shuffle: bool = True,
    ) -> None:
        super().__init__(dataset)

        self.Item = self.fields[ITEM, ID]
        self.source = self.dataset.to_rows(
            {self.Item: list(range(self.Item.count))}
        )
        self.source = tuple(self.source)
        self.launcher = GroupLauncher(partition, shuffle=shuffle).sharding_filter()

    def __getstate__(self):
        r"""Return serialization state, avoiding expensive traversal."""
        # `traverse_dps' will be particularly time-consuming
        # if a lot of data is buffered.
        # Hence, we directly return the connected datapipes.
        state = self.__dict__
        if IterDataPipe.getstate_hook is not None:
            return self.launcher
        return state

    def __iter__(self):
        for i in self.launcher:
            yield self.source[i].copy()