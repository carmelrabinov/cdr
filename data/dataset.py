import cv2
import numpy as np
import os

import random
from tqdm import tqdm
import torch
from torch.utils import data
from typing import NamedTuple, List
import re

from data.data_augmentation import RandomColorJitter, RandomGaussianBlur, GlobalShift
from data.utils import load_label, load_video, seg_mask_transform, to_tensor, normalize
from data.video_sampler import VideoRandomSampler


class TwoStateActionPairIndex(NamedTuple):
    first_video_name: str
    first_path: str
    second_video_name: str
    second_path: str
    state: torch.FloatTensor
    next_state: torch.FloatTensor
    observation_index: int
    next_observation_index: int
    seg_mask_path: str
    action: torch.FloatTensor

    def flip_videos(self):
        video_name = self.first_video_name
        video_path = self.first_path
        return self._replace(first_video_name=self.second_video_name,
                             first_path=self.second_path,
                             second_video_name=video_name,
                             second_path=video_path)


class ControlDataset(data.Dataset):
    """ a Dataset object for video sequences of controlled object"""
    def __init__(self, input_dir_path, n_videos, sample_range, config, load_seg_masks=False):

        # action state representation
        self.action_files_sfx = config["action_files_sfx"]
        self.second_video_sfx = config["second_video_sfx"]
        self.n_videos_samples = n_videos
        self.sample_range = sample_range
        self.add_video_name = False
        self.input_dir_path = input_dir_path
        self.use_oracle_states = config["use_oracle_states"]

        self.fix_action_bug = config["fix_action_bug"] if "fix_action_bug" in config else False
        self.image_size = config["image_size"] if "image_size" in config else 128

        # CDR
        self.contrastive_dr = config["contrastive_dr"]
        self.use_both_textures = config["use_both_textures"]
        self.contrastive_and_mse = config["contrastive_and_mse"] if "contrastive_and_mse" in config else False

        # Data Augmentations
        self._use_data_aug = config["use_data_aug"]
        self._color_jitter_p: config["color_jitter_p"]
        self._gaussian_blur_p: config["gaussian_blur_p"]
        self._contrastive_data_aug = config["contrastive_data_aug"]
        if self._use_data_aug:
            self.aumentations_dict = {config["color_jitter_p"]: RandomColorJitter(),
                                      config["gaussian_blur_p"]: RandomGaussianBlur(),
                                      config["global_shift_p"]: GlobalShift()
                                      }

        # segmentation_masks, for calculating IOU
        self.load_seg_masks = load_seg_masks
        self.seg_mask_sfx = config["seg_masks_files_sfx"] if "seg_masks_files_sfx" in config else "_seg_mask.npy"

        # load video list
        files = os.listdir(input_dir_path)
        video_list = [f for f in files if re.match("video_[0-9]+.npy", f)]
        if sample_range != -1:
            video_list_ = video_list
            video_list = [f for f in video_list_ if int(re.findall(r'\d+', f)[0]) in range(sample_range[0], sample_range[1])]
        print(f"Found {len(video_list)} files in sample rage {sample_range} to load")

        # check for max num of videos
        video_list = sorted(video_list)
        if n_videos > 0 and len(video_list) > n_videos:
            video_list = sorted(video_list)[:n_videos]
        print(f"Loading only {len(video_list)} files")

        self.video_list = video_list
        self.data_index = self.generate_pairs(video_list)
        assert len(self.data_index) != 0, "No data was selected"

    def image_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        if np.max(img) > 1:
            img = img / 255.
        if self.image_size != 128:
            img = cv2.resize(img, (self.image_size, self.image_size))
        return to_tensor(img).float()

    def augment_2_images(self, img_1: torch.FloatTensor, img_2: torch.FloatTensor) -> [torch.FloatTensor, torch.FloatTensor]:

        # apply different transform to observation and next_observation
        if self._contrastive_data_aug:
            return self.augment(img_1.unsqueeze(0)).squeeze(0), self.augment(img_2.unsqueeze(0)).squeeze(0)

        # apply the same transform to observation and next_observation
        else:
            augmented = self.augment(torch.cat((img_1.unsqueeze(0), img_2.unsqueeze(0)), dim=0))
            return augmented[0], augmented[1]

    def augment(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """ apply augmentations to batch of images
            expect img to be with dimension (b,c,h,w) """
        for p, aug in self.aumentations_dict.items():
            if random.uniform(0, 1) <= p:
                img = aug.augment(img)
        return img

    def generate_pairs(self, video_list):
        # load existing videos
        data_index = []
        missing_videos = []
        video_count = 0
        frames_count = 0
        self.video2frames = {}

        print("Generating a Texture dataset\n")
        for vid_idx, video_name in enumerate(tqdm(video_list, desc="Loading")):

            video_path = os.path.join(self.input_dir_path, video_name)
            second_video_name = video_name[:-4] + self.second_video_sfx + ".npy"
            second_video_path = os.path.join(self.input_dir_path, second_video_name)
            label_path = video_path[:-4]+self.action_files_sfx
            seg_mask_path = video_path[:-4] + self.seg_mask_sfx
            if not os.path.exists(video_path) or not os.path.exists(label_path):
                missing_videos.append(video_name)
                continue
            if (self.contrastive_dr or self.use_both_textures) and not os.path.exists(second_video_path):
                missing_videos.append(video_name)
                continue
            if self.load_seg_masks and not os.path.exists(seg_mask_path):
                missing_videos.append(video_name)
                continue
            elif not os.path.exists(seg_mask_path):
                seg_mask_path = ""

            vid = load_video(video_path)
            label = load_label(label_path)
            self.video2frames[vid_idx] = []
            for i in range(len(vid) - 1):
                pair_index = TwoStateActionPairIndex(first_video_name=video_name,
                                                     first_path=video_path,
                                                     second_video_name=second_video_name,
                                                     second_path=second_video_path,
                                                     state=torch.from_numpy(label["state"][i]),
                                                     next_state=torch.from_numpy(label["next_state"][i]),
                                                     observation_index=i,
                                                     next_observation_index=i+1,
                                                     seg_mask_path=seg_mask_path,
                                                     action=torch.from_numpy(label["action"][i]),
                                                     )
                data_index.append(pair_index)
                self.video2frames[vid_idx].append(frames_count)
                frames_count += 1
            video_count += 1

        if missing_videos:
            print(f"Missing Videos: {missing_videos}")
        else:
            print("Finished loading all files")
        print(f"Dataset contains {len(data_index)} (state, next_state, action) pairs from {video_count} videos\n")
        return data_index

    def get_item_default(self, index):
        """return (observation, next_observation, action) tuples"""
        pair_index: TwoStateActionPairIndex = self.data_index[index]

        # use oracle states as observations ("perfect representations")
        if self.use_oracle_states:
            observation = pair_index.state
            next_observation = pair_index.next_state
            action = pair_index.action

        # use image as observations
        else:
            if self.contrastive_dr or self.use_both_textures:
                # with probability of 0.5 sample the texture to predict from
                if np.random.rand() < 0.5:
                    pair_index = pair_index.flip_videos()

            first_video = load_video(pair_index.first_path)
            assert first_video.any(), f"Error in video {pair_index.first_path}"

            if self.contrastive_dr:
                second_video = load_video(pair_index.second_path)
                assert first_video.any(), f"Error in video {pair_index.second_path}"
            else:
                second_video = first_video

            observation = self.image_to_tensor(first_video[pair_index.observation_index])
            next_observation = self.image_to_tensor(second_video[pair_index.next_observation_index])
            action = pair_index.action

            # data augmentations:
            # apply different augmentations to observation and next_observation if contrastive_augmentations is true
            # else apply the same augmentations to observation and next_observation
            if self._use_data_aug:
                observation, next_observation = self.augment_2_images(observation, next_observation)

            # perform ResNet normalization
            observation = normalize(observation)
            next_observation = normalize(next_observation)

        if self.load_seg_masks:
            seg_mask_ = load_video(pair_index.seg_mask_path).astype(bool)
            seg_mask = seg_mask_transform(seg_mask_[pair_index.observation_index])
            next_seg_mask = seg_mask_transform(seg_mask_[pair_index.next_observation_index])
        else:
            seg_mask = []
            next_seg_mask = []
        return (observation, next_observation, action), (seg_mask, next_seg_mask),  pair_index

    def get_item_contrstive_and_mse(self, index):
        pair_index: TwoStateActionPairIndex = self.data_index[index]

        first_video = load_video(pair_index.first_path)
        assert first_video.any(), f"Error in video {pair_index.first_path}"

        second_video = load_video(pair_index.second_path)
        assert first_video.any(), f"Error in video {pair_index.second_path}"

        observation = normalize(self.image_to_tensor(first_video[pair_index.observation_index]))
        observation_tag = normalize(self.image_to_tensor(second_video[pair_index.observation_index]))
        next_observation = normalize(self.image_to_tensor(first_video[pair_index.next_observation_index]))
        next_observation_tag = normalize(self.image_to_tensor(second_video[pair_index.next_observation_index]))
        action = pair_index.action

        if self.load_seg_masks:
            seg_mask_ = load_video(pair_index.seg_mask_path).astype(bool)
            seg_mask = seg_mask_transform(seg_mask_[pair_index.observation_index])
            next_seg_mask = seg_mask_transform(seg_mask_[pair_index.next_observation_index])
        else:
            seg_mask = []
            next_seg_mask = []
        return (observation, observation_tag, next_observation, next_observation_tag, action), (seg_mask, next_seg_mask),  pair_index

    def __getitem__(self, index):
        if self.contrastive_and_mse:
            return self.get_item_contrstive_and_mse(index)
        else:
            return self.get_item_default(index)

    def __len__(self):
        return len(self.data_index)


def get_dataloader(input_dir: str, n_videos: int, sample_range, config,
                   workers: int, load_seg_masks: bool, use_full_batch_size: bool = False) -> data.DataLoader:
    """
    Generate a DataLoader
    :param input_dir: path to directory which holds the data
    :param n_videos: number of videos to load
    :param sample_range: range of video indexes to load, used to separate between training and validation data from the same directory
    :param config: config dictionary
    :param workers:
    :param load_seg_masks:
    :param use_full_batch_size:
    :return:
    """
    dataset = ControlDataset(input_dir_path=input_dir,
                             n_videos=n_videos,
                             sample_range=sample_range,
                             config=config,
                             load_seg_masks=load_seg_masks)

    # each batch contains n_samples_from_each_video
    if config["local_video_sampling"]:
        video_sampler = VideoRandomSampler(dataset.video2frames, config["n_samples_from_each_video"])
        shuffle = False
    # each batch contains random samples
    else:
        video_sampler = None
        shuffle = True

    batch_size = len(dataset) if use_full_batch_size else config["batch_size"]
    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             sampler=video_sampler,
                             num_workers=workers,
                             drop_last=False,
                             pin_memory=True,)
    return loader
