import os

from config import IMG_DIR


def get_data():
    samples = []
    dirs = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
    print('len(dirs): ' + str(len(dirs)))

    return samples


if __name__ == "__main__":
    get_data()
