from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import myutils
import NN

TYPES_NAMES_TO_NUMBERS = {
    '1goal': 0,
    '2chance': 1,
    '3special': 2
}
SOURCE_PATH = 'Data/'
GRAY_SCALE = False
NUMBER_FRAMES_TO_EXTRACT = 100


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

        self.clips = ['{}/{}/'.format(d['gameID'],
                                     d['eventID']) for d in json_data]
        self.num_clips = len(json_data)
        self.types = [d["type"] for d in json_data]

    def adapt_clip(self, clip):
        frames = self.get_frames(clip)
        frames = frames / 255
        if GRAY_SCALE:
            frames = np.moveaxis(frames, 3, 0)
        return torch.tensor(frames.astype(np.float32))

    def get_frames(self, clip):
        ret = []
        for i in range(0, NUMBER_FRAMES_TO_EXTRACT, NN.FRAMES_STEP):
            img = cv2.imread('{}{}{}.jpg'.format(SOURCE_PATH, clip, i))
            #img = cv2.resize(img, (224, 224))
            img = np.moveaxis(img, 2, 0)

            if GRAY_SCALE:
                img = myutils.rgb2gray(img)
            ret.append(img)

        return np.array(ret)


def draw_image(dataset, image_idx):
    image, type = dataset.__getitem__(image_idx)
    print('Image dimensions: {}'.format(image.shape))
    print('Type: {}'.format(type))
    image = np.moveaxis(image.numpy(), 0, -1)
    print(image[0].shape)
    if GRAY_SCALE:
        plt.imshow(image[0], cmap='gray')
    else:
        plt.imshow(image[0])
    plt.show()


if __name__ == "__main__":
    dataset = VideoDataset('test.json')
    draw_image(dataset, 0)
