from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import random

import myutils

TYPES_NAMES_TO_NUMBERS = {
    '1goal': 0,
    '2chance': 1,
    '3special': 2
}
SOURCE_PATH = 'Data/'
GRAY_SCALE = False
RESIZE = False
NUMBER_EXTRACTED_FRAMES = 100
FRAMES_STEP = 5


class VideoDataset(Dataset):
    def __init__(self, source_file):
        self.source_file = source_file
        self.num_clips = 0
        self.clips = []
        self.types = []

        self.clips_init_dataset()

    def __len__(self):
        return self.num_clips

    def __getitem__(self, item):
        return self.adapt_clip(self.clips[item]), TYPES_NAMES_TO_NUMBERS[self.types[item]]

    def clips_init_dataset(self):
        json_data = myutils.get_data(self.source_file)

        json_data = self.check_data_present(json_data)

        self.clips = ['{}/{}/'.format(d['gameID'],
                                     d['eventID']) for d in json_data]
        self.num_clips = len(json_data)
        self.types = [d["type"] for d in json_data]

    def check_data_present(self, data):
        ret = []

        for d in data:
            if os.path.exists('{}/{}/{}'.format(SOURCE_PATH, d['gameID'], d['eventID'])):
                ret.append(d)

        return ret

    def adapt_clip(self, clip):
        frames = self.get_frames(clip, (True if random.random() > 0.5 else False))
        frames = frames / 255
        if GRAY_SCALE:
            frames = np.moveaxis(frames, 3, 0)
        return torch.tensor(frames.astype(np.float32))

    def get_frames(self, clip, flip):
        ret = []
        for i in range(0, NUMBER_EXTRACTED_FRAMES, FRAMES_STEP):
            img = cv2.imread('{}{}{}.png'.format(SOURCE_PATH, clip, i))

            if flip:
                img = cv2.flip(img, 1)

            if RESIZE:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

            img = np.moveaxis(img, 2, 0)

            if GRAY_SCALE:
                img = myutils.rgb2gray(img)
            ret.append(img)

        return np.array(ret)


def draw_image(dataset, image_idx):
    image, type = dataset.__getitem__(image_idx)
    print('Image dimensions: {}'.format(image.shape))
    print('Type: {}'.format(type))
    image = np.moveaxis(image.numpy(), 1, -1)
    print(image[0].shape)
    if GRAY_SCALE:
        plt.imshow(image[0], cmap='gray')
    else:
        plt.imshow(image[0])
    plt.show()


if __name__ == "__main__":
    dataset = VideoDataset('test.json')
    draw_image(dataset, 0)
