import os

from tqdm import tqdm

from config import IMG_DIR


def get_data():
    samples = []
    dirs = [os.path.join(IMG_DIR, d) for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
    for dir in tqdm(dirs):
        files = [f for f in os.listdir(dir) if f.endswith('.jpg')]

        for f in files:
            img_path = os.path.join(dir, f)
            samples.append(img_path)

    return samples


if __name__ == "__main__":
    num_total = 10000
    num_same = int(num_total / 2)
    num_not_same = num_total - num_same

    samples = get_data()
    print(len(samples))
    print(samples[:10])

    for _ in range(num_same):
        None

    for _ in range(num_not_same):
        None
