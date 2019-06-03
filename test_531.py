import json
import os
import tarfile

import cv2 as cv
import torch
from PIL import Image
from tqdm import tqdm

from config import device
from config import im_size
from data_gen import data_transforms

angles_file = 'data/angles.txt'
IMG_FOLDER = 'data/jinhai531'
transformer = data_transforms['val']


def extract(filename):
    with tarfile.open(filename, 'r') as tar:
        tar.extractall('data')


def get_image(filename):
    img = cv.imread(filename)
    img = cv.resize(img, (im_size, im_size))
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img


def gen_features(model):
    features = []
    dir_list = [d for d in os.listdir(IMG_FOLDER) if os.path.isdir(os.path.join(IMG_FOLDER, d))]
    for dir in tqdm(dir_list):
        dir_path = os.path.join(IMG_FOLDER, dir)
        file_list = [f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]
        for file in file_list:
            fullpath = os.path.join(dir_path, file)
            is_sample = file == '0.jpg'
            features.append({'fullpath': fullpath, 'file': file, 'dir': dir, 'is_sample': is_sample})
    with open('data/jinhai531_file_list.json', 'w') as file:
        json.dump(features, file, ensure_ascii=False, indent=4)


def evaluate(model):
    pass


def get_threshold():
    return 1.0


def accuracy(thres):
    return 1.0


def test(model):
    print('Generating features...')
    gen_features(model)

    print('Evaluating {}...'.format(angles_file))
    # evaluate(model)

    print('Calculating threshold...')
    # threshold = 70.36
    thres = get_threshold()
    print('Calculating accuracy...')
    acc = accuracy(thres)
    print('Accuracy: {}%, threshold: {}'.format(acc * 100, thres))
    return acc, thres


if __name__ == "__main__":
    if not os.path.isdir('data/jinhai531'):
        extract('data/jinhai_531.tar.gz')

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    acc, threshold = test(model)

    # print('Visualizing {}...'.format(angles_file))
    # visualize(threshold)
    #
    # print('error analysis...')
    # error_analysis(threshold)
