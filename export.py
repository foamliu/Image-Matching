import time

import torch

from models import resnet50

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model'].module
    print(type(model))
    print('use_se: ' + str(model.use_se))
    # model.eval()

    filename = 'face-attributes.pt'
    torch.save(model.state_dict(), filename)
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))


    class HParams:
        def __init__(self):
            self.pretrained = False
            self.use_se = False


    config = HParams()

    print('loading {}...'.format(filename))
    start = time.time()
    model = resnet50(config)
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))