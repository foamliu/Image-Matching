import json
import math
import os
import pickle
import tarfile

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import device
from config import im_size
from data_gen import data_transforms

angles_file = 'data/angles.txt'
IMG_FOLDER = 'data/jinhai531'
pickle_file = 'data/jinhai531_features.pkl'
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
    data = []
    dir_list = [d for d in os.listdir(IMG_FOLDER) if os.path.isdir(os.path.join(IMG_FOLDER, d))]
    for dir in tqdm(dir_list):
        dir_path = os.path.join(IMG_FOLDER, dir)
        file_list = [f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]
        for file in file_list:
            fullpath = os.path.join(dir_path, file)
            is_sample = file == '0.jpg'
            data.append({'fullpath': fullpath, 'file': file, 'dir': dir, 'is_sample': is_sample})
    with open('data/jinhai531_file_list.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    file_count = len(data)

    batch_size = 128

    with torch.no_grad():
        for start_idx in tqdm(range(0, file_count, batch_size)):
            end_idx = min(file_count, start_idx + batch_size)
            length = end_idx - start_idx

            imgs = torch.zeros([length, 3, im_size, im_size], dtype=torch.float)
            for idx in range(0, length):
                i = start_idx + idx
                filepath = data[i]['fullpath']
                imgs[idx] = get_image(filepath)

            features = model(imgs.to(device)).cpu().numpy()
            for idx in range(0, length):
                i = start_idx + idx
                feature = features[idx]
                data[i]['feature'] = feature

    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)


def evaluate(model):
    pass


def get_threshold():
    return 25.50393648495902


def accuracy(threshold):
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    num_tests = 0
    wrong = 0

    samples = [f for f in data if f['is_sample']]
    photos = [f for f in data if not f['is_sample']]
    for sample in tqdm(samples):
        feature0 = sample['feature']
        x0 = feature0 / np.linalg.norm(feature0)
        ad_no_0 = sample['dir']
        for photo in photos:
            feature1 = photo['feature']
            x1 = feature1 / np.linalg.norm(feature1)
            ad_no_1 = photo['dir']
            cosine = np.dot(x0, x1)
            cosine = np.clip(cosine, -1, 1)
            theta = math.acos(cosine)
            angle = theta * 180 / math.pi

            type = int(ad_no_0 == ad_no_1)

            num_tests += 1
            if type == 1 and angle > threshold or type == 0 and angle <= threshold:
                wrong += 1

    print('num_tests: {}, wrong: {}.'.format(num_tests, wrong))

    accuracy = 1 - wrong / num_tests
    return accuracy


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
