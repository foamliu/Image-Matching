import math

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import ResNetMatchModel


def get_image(file):
    img = cv.imread(file)
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img


def get_feature(model, file):
    img = get_image(file)
    imgs = img.unsqueeze(dim=0)
    with torch.no_grad():
        output = model(imgs)
    feature = output[0].cpu().numpy()
    return feature / np.linalg.norm(feature)


if __name__ == "__main__":
    device = torch.device('cpu')
    threshold = 21.07971786746929

    filename = 'image_matching.pt'
    model = ResNetMatchModel()
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    x0 = get_feature(model, '0.jpg')
    x1 = get_feature(model, '6.jpg')

    cosine = np.dot(x0, x1)
    cosine = np.clip(cosine, -1, 1)
    theta = math.acos(cosine)
    theta = theta * 180 / math.pi

    print(theta <= threshold)
