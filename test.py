import os
import torch
import yaml
from options import Options
from model import AG_CVAEGAN
os.environ['TORCH_HOME'] = '../pretrained_models'

if __name__ == '__main__':

    opt = Options().parse()

    with open('config.yaml') as fp:
        param = yaml.safe_load(fp)

    # create a pseudo input: a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, 256, 1, 128, 128])

    model = AG_CVAEGAN(opt, test_input, param)

    model.train()


