import cv2
import numpy as np
from typing import List


class AttrDict(object):
    def __init__(self, attr):
        for k, v in attr.items():
            setattr(self, k, v)


def save_video_as_mp4(video: List[np.ndarray], save_path: str = 'simulation', fps: int = 10,
                      image_width: int = 128, image_hight: int = 128) -> None:
    """
    save a stream of images as videos file
    :param video: a list of np.ndarray frames
    :param save_path: path to save video
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 0x00000021
    vid = cv2.VideoWriter(save_path + ".mp4", fourcc, fps, (image_width, image_hight))
    for frame in video:
        vid.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vid.release()


def save_video_as_npy(video: List[np.ndarray], save_path: str = 'simulation') -> None:
    """
    save a stream of images as videos file
    :param video: a list of np.ndarray frames
    :param save_path: path to save video
    """
    np.save(save_path, np.array(video))
