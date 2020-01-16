import os
import pickle
import random

from PIL import Image
from torchvision import transforms

from config import pickle_file

old_folder = 'data/data/frame/cron20190326'


def image_aug(split, transformer):
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
        img.save('images/{}_aug_{}.jpg'.format(split, i))


def main():
    transformer = transforms.Compose([
        transforms.ColorJitter(brightness=0.250, contrast=0.250, saturation=0.250, hue=[-0.2, 0.2]),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Resize((224, 224)),
    ])
    image_aug('train', transformer)

    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    image_aug('valid', transformer)


if __name__ == "__main__":
    main()
