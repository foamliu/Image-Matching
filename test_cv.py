import os
import time

import cv2
from tqdm import tqdm

from config import num_tests
from match_feature import Recognition

TEST_FILE = 'data/test_pairs_rectified.txt'
IMG_FOLDER = 'data/data/frame/cron20190326/'


def test():
    with open(TEST_FILE) as file:
        lines = file.readlines()

    wrong = 0
    num_tests = 0
    num_ex = 0
    for line in tqdm(lines):
        tokens = line.split()
        imagepath1 = os.path.join(IMG_FOLDER, tokens[0])
        imagepath2 = os.path.join(IMG_FOLDER, tokens[1])
        type = int(tokens[2])

        image1 = cv2.imread(imagepath1)
        image1 = cv2.resize(image1, (720, 960))
        image2 = cv2.imread(imagepath2)

        try:
            res, inv = Recognition(image1, image2)

            num_tests += 1
            if type == 1 and res != 'ok' or type == 0 and res == 'ok':
                wrong += 1
        except TypeError:
            num_ex += 1
            print(line)

    accuracy = 1 - wrong / num_tests
    return accuracy, num_ex


if __name__ == "__main__":
    start = time.time()
    acc, num_ex = test()
    end = time.time()
    elapsed = end - start

    print('acc: {}, exceptions: {}'.format(acc, num_ex))
    print('elapsed: {}'.format(elapsed))
    print('elapsed per pair: {}'.format(elapsed / num_tests))
