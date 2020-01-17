import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
im_size = 224
channel = 3
emb_size = 512

# Training parameters
num_workers = 8  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
num_classes = 9934
num_samples = 373471
num_tests = 10000
DATA_DIR = 'data'
IMG_DIR = 'data/cron20190326_resized/'
IMG_DIR_TEST = 'data/jinhai531_resized'
# IMG_DIR = 'data/data/frame/cron20190326'
# IMG_DIR_TEST = 'data/test/data'
pickle_file = 'data/cron20190326.pickle'
pickle_test_file = 'data/jinhai_531.pickle'
