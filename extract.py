import os

from utils import extract_zip, extract_tgz

if __name__ == "__main__":
    if not os.path.isdir('data/data/frame/cron20190326/'):
        extract_zip('data/cron20190326.zip')
        extract_zip('data/cron20190415.zip')

    extract_tgz('data/jinhai_531.tar.gz')
