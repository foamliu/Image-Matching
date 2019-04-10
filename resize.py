import os
from multiprocessing import Pool

import cv2 as cv
from tqdm import tqdm

from config import IMG_DIR

image_w = 112
image_h = 112
folder = 'data/cron20190326_resized'


def resize_images(d):
    dir = os.path.join(IMG_DIR, d)
    files = [f for f in os.listdir(dir) if f.endswith('.jpg')]

    dst_folder = os.path.join(folder, d)
    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)

    for f in files:
        img_path = os.path.join(dir, f)
        img = cv.imread(img_path)
        img = cv.resize(img, (image_w, image_h))
        dst_file = os.path.join(dst_folder, f)
        cv.imwrite(dst_file, img)


if __name__ == "__main__":
    if not os.path.isdir(folder):
        os.makedirs(folder)

    dirs = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
    with Pool(6) as p:
        r = list(tqdm(p.imap(resize_images, dirs), total=len(dirs)))
