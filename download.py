import hashlib
import os
import pickle
import posixpath
from subprocess import Popen, PIPE
from urllib.parse import urlsplit, unquote

from tqdm import tqdm


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def url2filename(url):
    urlpath = urlsplit(url).path
    basename = posixpath.basename(unquote(urlpath))
    if (os.path.basename(basename) != basename or
            unquote(posixpath.basename(urlpath)) != basename):
        raise ValueError  # reject '%2f' or 'dir%5Cbasename.ext' on Windows
    return basename


def download_rename(url):
    # print(url)
    m = hashlib.md5()
    m.update(bytes(url, encoding='utf-8'))
    filename = '{}.jpg'.format(m.hexdigest())

    process = Popen(["wget", '-N', url, "-O", os.path.join(folder, filename)], stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    return filename


def download(url):
    process = Popen(["wget", '-N', url, "-P", folder], stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()


if __name__ == "__main__":
    filename = 'data/photos.csv'

    with open(filename, 'r') as fp:
        lines = fp.readlines()

    folder = 'data/gumin_116'
    ensure_folder(folder)

    samples = []

    for line in tqdm(lines):
        tokens = line.split(',')
        plan_id = tokens[0].strip().replace('"', '')
        photo_url = tokens[1].strip().replace('"', '')
        photo_filename = download(photo_url)

        sample_url = tokens[2].strip().strip('"')
        sample_filename = download_rename(sample_url)

        samples.append({'plan_id': plan_id, 'photo_filename': photo_filename, 'sample_filename': sample_filename})
        break

    print(samples[0])

    with open('data/gumin_116.pkl', 'wb') as fp:
        pickle.dump(samples, fp)
