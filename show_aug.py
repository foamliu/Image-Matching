import os
import pickle
import random

from PIL import Image
from torchvision import transforms

from config import pickle_file

old_folder = 'data/data/frame/cron20190326'

if __name__ == "__main__":
    transformer = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ])

    with open(pickle_file, 'rb') as fp:
        samples = pickle.load(fp)

    samples = random.sample(samples, 10)

    for i, sample in enumerate(samples):
        img_path = sample['img_path']
        full_path = os.path.join(old_folder, img_path)
        print(full_path)
        img = Image.open(full_path)
        print(img.size)
        img = transformer(img)
        img.save('images/train_aug_{}.jpg'.format(i))

    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    with open(pickle_file, 'rb') as fp:
        samples = pickle.load(fp)

    samples = random.sample(samples, 10)

    for i, sample in enumerate(samples):
        img_path = sample['img_path']
        full_path = os.path.join(old_folder, img_path)
        print(full_path)
        img = Image.open(full_path)
        print(img.size)
        img = transformer(img)
        img.save('images/valid_aug_{}.jpg'.format(i))
