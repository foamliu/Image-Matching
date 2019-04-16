import os

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
        image1 = tokens[0]
        image1 = image1[:image1.index('/')]
        image1 = os.path.join(IMG_FOLDER, image1)
        image2 = os.path.join(IMG_FOLDER, tokens[1])
        type = int(tokens[2])

        res, inv = Recognition(image1, image2)

        if type == 1 and res != 'ok' or type == 0 and res == 'ok':
            wrong += 1

    accuracy = 1 - wrong / num_tests
    return accuracy


if __name__ == "__main__":
    acc = test()

    print('acc: {}'.format(acc))
