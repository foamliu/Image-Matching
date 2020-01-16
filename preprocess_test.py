import os
import random

from tqdm import tqdm

from config import IMG_DIR_TEST, num_tests


def pick_one_file(folder):
    files = [f for f in os.listdir(os.path.join(IMG_DIR_TEST, folder)) if
             f.endswith('.jpg') and not f.endswith('0.jpg')]
    file = random.choice(files)
    file = os.path.join(folder, file)
    file = file.replace('\\', '/')
    return file


if __name__ == "__main__":
    VOCAB = {}
    IVOCAB = {}

    num_same = int(num_tests / 2)
    num_not_same = num_tests - num_same

    out_lines = []

    dirs = [d for d in os.listdir(IMG_DIR_TEST) if os.path.isdir(os.path.join(IMG_DIR_TEST, d))]

    for _ in tqdm(range(num_same)):
        folder = random.choice(dirs)
        while len([f for f in os.listdir(os.path.join(IMG_DIR_TEST, folder)) if
                   f.endswith('.jpg') and not f.endswith('0.jpg')]) < 1:
            folder = random.choice(dirs)

        files = [f for f in os.listdir(os.path.join(IMG_DIR_TEST, folder)) if
                 f.endswith('.jpg') and not f.endswith('0.jpg')]
        file_1 = random.choice(files)
        file_0 = os.path.join(folder, '0.jpg').replace('\\', '/')
        file_1 = os.path.join(folder, file_1).replace('\\', '/')
        out_lines.append('{} {} {}\n'.format(file_0, file_1, 1))

    for _ in tqdm(range(num_not_same)):
        folders = random.sample(dirs, 2)
        while len([f for f in os.listdir(os.path.join(IMG_DIR_TEST, folders[0])) if
                   f.endswith('.jpg') and not f.endswith('0.jpg')]) < 1 or len(
            [f for f in os.listdir(os.path.join(IMG_DIR_TEST, folders[1])) if
             f.endswith('.jpg') and not f.endswith('0.jpg')]) < 1:
            folders = random.sample(dirs, 2)

        file_0 = folders[0] + '/' + '0.jpg'
        file_1 = pick_one_file(folders[1])
        out_lines.append('{} {} {}\n'.format(file_0, file_1, 0))

    with open('data/test_pairs.txt', 'w') as file:
        file.writelines(out_lines)
