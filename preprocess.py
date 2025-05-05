import json
import math
import os
import cv2
import random
import numpy as np

import myutils

FRAMES_PER_SEC = 30
DOWNSCALE_FACTOR = 4
NUMBER_FRAMES_TO_EXTRACT = 20

URL_PREFIX = "https://vipr-app.s3.amazonaws.com/data/"
TEMP_FILE_NAME = "temp-clip.mp4"

SOURCE_FILE = 'Video jsons/videos2024.json'
FAILED_FILE = 'failed.json'
PROCESSED_FILE = 'data.json'
DATA_PATH = 'Data/'

TEST_FILE = 'test.json'
TRAIN_FILE = 'train.json'


def inaccessible(response):
    if response.status_code == 200:
        return False
    return True


def download_file(response):
    with open(TEMP_FILE_NAME, 'wb') as file:
        file.write(response.content)


def make_folders(data):
    os.makedirs(DATA_PATH + str(data['gameID']), exist_ok=True)
    os.makedirs(DATA_PATH + str(data['gameID']) + '/' + str(data['eventID']), exist_ok=True)
    return DATA_PATH + str(data['gameID']) + '/' + str(data['eventID'])


def save_frames(data):
    path = make_folders(data)
    print('.', end='')

    video_clip = cv2.VideoCapture(URL_PREFIX + data['file'])
    if ((not video_clip.isOpened()) or
            video_clip.get(cv2.CAP_PROP_FRAME_HEIGHT) != myutils.ORIGINAL_IMAGE_HEIGHT or
            video_clip.get(cv2.CAP_PROP_FRAME_WIDTH) != myutils.ORIGINAL_IMAGE_WIDTH):
        return True

    frames = []
    while True:
        ret, frame = video_clip.read()

        if not ret:
            break
        frames.append(frame)

    video_clip.release()
    cv2.destroyAllWindows()
    print('.', end='')

    if len(frames) < 15 * FRAMES_PER_SEC:
        return True

    print('.', end='')

    step = len(frames) / NUMBER_FRAMES_TO_EXTRACT
    i, current_duration = 0, 0.

    while i < NUMBER_FRAMES_TO_EXTRACT:
        frame = myutils.downscale_frames(frames[int(math.floor(current_duration))],
                                         myutils.ORIGINAL_IMAGE_HEIGHT // DOWNSCALE_FACTOR,
                                         myutils.ORIGINAL_IMAGE_WIDTH // DOWNSCALE_FACTOR)
        cv2.imwrite(path + '/' + str(i) + '.jpg', frame)

        i += 1
        current_duration = i * step
        
    print('.', end='')

    return False


def process_data(d):
    print('{}-{}'.format(d['gameID'], d['eventID']), end=' ')

    save_failed = save_frames(d)

    if save_failed:
        failed.append(d)
        print('failed')
    else:
        processed.append(d)
        print('done')


def divide_data(data):
    test = []
    train = []

    for i in range(len(data)):
        if i % 5 == 0:
            test.append(data[i])
        else:
            train.append(data[i])

    myutils.save_data(test, TEST_FILE)
    myutils.save_data(train, TRAIN_FILE)


if __name__ == '__main__':
    failed = myutils.get_data(FAILED_FILE)
    processed = myutils.get_data(PROCESSED_FILE)

    data = myutils.get_data(SOURCE_FILE)
    random.shuffle(data)

    skip = []

    for i in range(len(data)):
        if len(processed) == 30000:
            break

        d = data[i]
        if d in skip or d in processed or d in failed:
            continue
        process_data(d)

        if i % 10 == 0:
            print(len(processed))
            myutils.save_data(processed, PROCESSED_FILE)
            myutils.save_data(failed, FAILED_FILE)

    print('save')
    myutils.save_data(processed, PROCESSED_FILE)
    myutils.save_data(failed, FAILED_FILE)

    random.shuffle(processed)
    divide_data(processed)
