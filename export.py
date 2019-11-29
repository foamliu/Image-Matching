import time

import torch

from mobilenet_v2 import MobileNetV2
from models import ArcMarginModel

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model'].module
    metric_fc = checkpoint['metric_fc'].module

    filename = 'image_matching_mobile.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    # print(model)

    print('loading {}...'.format(filename))
    start = time.time()
    model = MobileNetV2()
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))

    # filename = 'metric_fc.pt'
    # print('saving {}...'.format(filename))
    # start = time.time()
    # torch.save(metric_fc.state_dict(), filename)
    # print('elapsed {} sec'.format(time.time() - start))
    #
    # print('loading {}...'.format(filename))
    # start = time.time()
    # metric_fc = ArcMarginModel()
    # metric_fc.load_state_dict(torch.load(filename))
    # print('elapsed {} sec'.format(time.time() - start))
