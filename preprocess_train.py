import os

from tqdm import tqdm

from config import IMG_DIR


def get_data():
    samples = []
    dirs = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
    print('len(dirs): ' + str(len(dirs)))

    for d in tqdm(dirs):
        build_mapping(d)

        dir = os.path.join(IMG_DIR, d)
        files = [f for f in os.listdir(dir) if f.lower().endswith('.jpg')]

        for f in files:
            img_path = os.path.join(d, f)
            samples.append({'img_path': img_path, 'label': dir2label[d]})

    return samples


def build_mapping(dir):
    global dir2label, label2dir
    if not dir in dir2label:
        next_index = len(dir2label)
        dir2label[dir] = next_index
        label2dir[next_index] = dir


if __name__ == "__main__":
    dir2label = {}
    label2dir = {}

    samples = get_data()
    print('len(samples): ' + str(len(samples)))
