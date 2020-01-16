import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR, pickle_file

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.250, contrast=0.250, saturation=0.250, hue=[-0.2, 0.2]),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class FrameDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data
        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = os.path.join(IMG_DIR, sample['img_path'])
        label = sample['label']

        # print(filename)
        img = Image.open(filename)
        img = self.transformer(img)

        # print(img.size())

        return img, label

    def __len__(self):
        return len(self.samples)
