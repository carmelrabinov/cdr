import re
import shutil

import torch
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import cv2


action_files_sfx = "_state_action_label"
second_video_sfx = "_2"
segmentation_mask_sfx = "_seg_mask"


def resize(x, size: int = 128):
    return transforms.Resize(size)(x)


def to_tensor(x):
    return transforms.ToTensor()(x)


def normalize(x):
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)


def image_transform(img, resize_to=128):
    if np.max(img) > 1:
        img = img/255.
    if resize_to != 128:
        img = cv2.resize(img, (resize_to, resize_to))
    return normalize(to_tensor(img))


def seg_mask_transform(mask):
    return transforms.ToTensor()(mask)


# TODO: update according to the new env
def im2pos_coordinates(pix_x, pix_z):
    """move from image pixels coordinates to world coordinates"""
    # x_lim = [-0.85, 0.86]
    # z_lim = [-1.22, 0.47]
    x_lim = [-0.365, 0.365]
    z_lim = [-0.95, -0.24]
    x = x_lim[1] - (x_lim[1] - x_lim[0]) * pix_x/127
    z = z_lim[1] - (z_lim[1] - z_lim[0]) * pix_z/127
    return x, z


# TODO: update according to the new env
def pos2im_coordinates(x, z):
    """move from world coordinates to image pixels coordinates"""
    # x_lim = [-0.85, 0.86]
    # z_lim = [-1.22, 0.47]
    x_lim = [-0.365, 0.365]
    z_lim = [-0.95, -0.24]

    pix_x = int(127 * (x_lim[1] - x) / (x_lim[1] - x_lim[0]))
    pix_z = int(127 * (z_lim[1] - z) / (z_lim[1] - z_lim[0]))
    return pix_x, pix_z


def add_arrow_to_image(image, action):
    """
    add action arrow to image
    :param image: image of the observation
    :param action: action in (x_source, y_source) (x_target, y_target) formant
    :return:
    """
    x_tail, z_tail = pos2im_coordinates(action[0], action[1])
    x_head, z_head = pos2im_coordinates(action[2], action[3])

    # visual params
    color = (0, 255, 0)
    thickness = 3

    return cv2.arrowedLine(image, (x_tail, z_tail), (x_head, z_head), color, thickness)


def load_label(path: str) -> dict:
    """load label file in the right format"""
    if not os.path.exists(path):
        print(f"Warning, try to load non-exist label {path}")
        return None
    return np.load(path, allow_pickle=True).tolist()


def process_action(label_dict: dict) -> torch.Tensor:
    """extract action from label and transform to Tensor"""
    target_pos = np.array(label_dict["action"], dtype=float)[:, :3]
    pos = np.array(label_dict["ee_positions"], dtype=float)
    action = torch.from_numpy(np.concatenate((pos, target_pos), axis=-1))
    # label["collisions"] = torch.from_numpy(np.array(label_dict["collisions"]))
    return action


def load_single_image(path: str) -> np.uint8:
    """load to single image file"""
    if not os.path.exists(path):
        print(f"Warning, try to load non-exist image {path}")
        return None
    if path.endswith(".npy"):
        img = np.load(path)
    elif path.endswith(".png") or path.endswith(".jpeg") or path.endswith(".jpg"):
        img = plt.imread(path)
        if img.dtype != "uint8":
            img = (255 * img).astype(np.uint8)
    return img


def load_video(path: str) -> np.ndarray:
    if not os.path.exists(path):
        print(f"Warning, try to load non-exist file {path}")
        return None
    return np.load(path)


def load_seg_masks_from_video(path: str, frame_index: int = -1) -> np.ndarray:
    """load a full trajectory segmentation mask and return full mask or single frame"""
    seg_mask = load_video(path)
    if frame_index >= 0 and seg_mask is not None:
        return seg_mask[frame_index]
    return seg_mask


def load_frame_from_video(path: str, frame_index: int) -> np.ndarray:
    """load a full trajectory video file and return a single frame from it"""
    vid = load_video(path)
    img = vid[frame_index]
    return img


def visualize_trajectory(path, plot_segmentation=False, save_path=None) -> None:
    ref_video = load_video(path)
    label_path = path[:-4] + action_files_sfx + ".npy"
    label = load_label(label_path)
    actions = label["action"]

    if plot_segmentation:
        ref_seg_mask = load_video(path[:-4] + segmentation_mask_sfx + ".npy")

    n_cols = len(ref_video)
    n_rows = 2 if plot_segmentation else 1
    fig = plt.figure(figsize=(3*n_cols, 3*n_rows))
    fig.suptitle(path, fontsize=12)
    for i in range(n_cols):
        ref_image = add_arrow_to_image(np.copy(ref_video[i]), actions[i])

        fig.add_subplot(n_rows, n_cols, i+1).set_title(f"{i}", fontsize=20)
        plt.imshow(ref_image)
        plt.axis('off')

        if plot_segmentation:
            ref_mask = ref_seg_mask[i]
            fig.add_subplot(n_rows, n_cols, n_cols + i+1).set_title(f"segmentation {i}", fontsize=20)
            plt.imshow(ref_mask[:, :, 0], cmap=plt.cm.gray)
            plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path + ".jpg")
    plt.show()


def videos_to_images(dir_path: str, load_segmantation_masks: bool = False) -> None:
    """
    load a full trajectory video file (and optionally segmentation mask trajectory)
    and save each image from it as a separate file
    """
    os.makedirs(dir_path + "_processed", exist_ok=True)
    videos = [v for v in os.listdir(dir_path) if re.match("video_[0-9]+.npy", v)]
    for video_path in videos:
        video = load_video(os.path.join(dir_path, video_path))
        for i in range(len(video)):
            im = Image.fromarray(video[i].astype(np.uint8))
            im.save(dir_path + f'_processed/{video_path[:-4]}_{i}.png')
        if load_segmantation_masks:
            seg_mask = load_seg_masks_from_video(os.path.join(dir_path, video_path[:-4] + segmentation_mask_sfx + ".npy"))
            for i in range(len(video)):
                np.save(dir_path + f'_processed/{video_path[:-4]}_{i}_seg_mask', seg_mask[i])


if __name__ == '__main__':
    # videos_to_images("D:/Representation_Learning/datasets/textured_rope_new_ood", load_seg=True)
    # process_rope_dataset("/mnt/data/carmel_data/datasets/textured_rope_val_masks")
    visualize_trajectory(r"D:\Representation_Learning\datasets\cube_2d_states_textures\video_51.npy", plot_segmentation=False)
    # fix_action_bug_rope_dataset()
    # videos_to_images("/mnt/data/carmel_data/datasets/textured_rope_val_masks_1", load_seg=True, )