import json
import numpy as np


ORIGINAL_IMAGE_WIDTH = 1280
ORIGINAL_IMAGE_HEIGHT = 720

KEYS = [
    "sport",
    "type",
    "preroll",
    "postroll"
]


def save_data(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def read_data(file_name):
    with open(file_name) as f:
        data = json.load(f)
        if isinstance(data, dict) and "data" in data.keys():
            return data["data"]
        return data


def get_filtered_data(file_name, s=None, t=None, pre=None, post=None):
    ret = []
    for line in read_data(file_name):
        if ((KEYS[0] not in line.keys() or s is None or line[KEYS[0]] == s) and
                (t is None or line[KEYS[1]] == t) and
                (KEYS[2] not in line.keys() or pre is None or line[KEYS[2]] == pre) and
                (KEYS[3] not in line.keys() or post is None or line[KEYS[3]] == post)):
            ret.append(line)
    return ret


def get_data(file_name):
    return get_filtered_data(file_name, s="Football", pre=20)


def downscale_frames(frame, new_height, new_width):
    original_height = frame.shape[0]
    original_width = frame.shape[1]
    return frame.reshape((
        new_height, 
        original_height // new_height,
        new_width, 
        original_width // new_width,
        3
        )).mean(3).mean(1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
