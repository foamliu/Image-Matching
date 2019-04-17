import os

import cv2
from tqdm import tqdm

from config import num_tests
from match_feature import Recognition

TEST_FILE = 'data/test_pairs.txt'
IMG_FOLDER = 'data/data/frame/cron20190326'


def test():
    with open(TEST_FILE) as file:
        lines = file.readlines()

    wrong = 0
    for line in tqdm(lines):
        tokens = line.split()
        imagepath1 = tokens[0]
        imagepath1 = imagepath1[:imagepath1.index('/')] + '\\0.jpg'
        imagepath1 = os.path.join(IMG_FOLDER, imagepath1)
        imagepath1 = imagepath1.replace('/', '\\')
        imagepath2 = os.path.join(IMG_FOLDER, tokens[1])
        type = int(tokens[2])

        image1 = cv2.imread(imagepath1)
        image1 = cv2.resize(image1, (720, 960))
        image2 = cv2.imread(imagepath2)

        # try:
        res, inv = Recognition(image1, image2)
        # except TypeError:
        #     res, inv = 'ok', 0

        if type == 1 and res != 'ok' or type == 0 and res == 'ok':
            wrong += 1

    accuracy = 1 - wrong / num_tests
    return accuracy


if __name__ == "__main__":
    acc = test()

    print('acc: {}'.format(acc))
