import os
import pickle
import random

from PIL import Image
from torchvision import transforms

from config import pickle_file, IMG_DIR

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
        full_path = os.path.join(IMG_DIR, img_path)
        print(full_path)
        img = Image.open(full_path)
        img = transformer(img)
        img.save('images/{}_aug.jpg'.format(i))
