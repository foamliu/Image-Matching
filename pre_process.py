import os
import pickle

from tqdm import tqdm

from config import IMG_DIR, pickle_file


def get_data():
    samples = []
    dirs = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
    for d in tqdm(dirs):
        build_vocab(d)

        dir = os.path.join(IMG_DIR, d)
        files = [f for f in os.listdir(dir) if f.endswith('.jpg')]

        for f in files:
            img_path = os.path.join(dir, f)
            img_path = img_path.replace('\\', '/')
            samples.append({'img': img_path, 'label': VOCAB[d]})

    return samples


def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


if __name__ == "__main__":
    VOCAB = {}
    IVOCAB = {}

    num_total = 10000
    num_same = int(num_total / 2)
    num_not_same = num_total - num_same

    samples = get_data()
    with open(pickle_file, 'wb') as file:
        pickle.dump(samples, file)

    print(len(samples))
    print(samples[:10])

    out_lines = []

    for _ in range(num_same):
        None

    for _ in range(num_not_same):
        None

    with open('data/test_pairs.txt', 'w') as file:
        file.writelines(out_lines)
