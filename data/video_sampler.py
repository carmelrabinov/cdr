import torch
from torch.utils.data.sampler import Sampler
import random


class VideoRandomSampler(Sampler):
    """
    sample n_samples_from_video in each batch
    """
    def __init__(self, video2frames: dict, n_samples_from_video:int = 2, replacement: bool = False, num_samples: bool = None):
        self.video2frames = video2frames
        self.n_samples_from_video = n_samples_from_video
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return sum([len(x) for x in self.video2frames.values()])
        return self._num_samples

    def __iter__(self):
        # generate chunks, each chunk is from the same video
        chunks = []
        for vid_indexs in self.video2frames.values():
            random.shuffle(vid_indexs)
            vid_chunks = [vid_indexs[x:x + self.n_samples_from_video] for x in range(0, len(vid_indexs), self.n_samples_from_video)]
            chunks.extend(vid_chunks)

        # shuffle chunks
        random.shuffle(chunks)

        # flatten chunks to one list
        index_list = []
        for chunk in chunks:
            index_list.extend(chunk)

        return iter(index_list)

    def __len__(self):
        return self.num_samples

