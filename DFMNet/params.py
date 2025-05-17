import torch
import torch.nn as nn
from mod.fusion import fusion_model

from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


import torch
import os
import torch.nn as nn
from thop import profile
from datasets.crowd_or import Crowd
import argparse

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default='/home/hlj/YYB/RGBT-datasetes/RGBTCCV3',
help='training data directory')
parser.add_argument('--dataset', default='RGBTCC')
args = parser.parse_args()


if __name__ == '__main__':

    datasets = Crowd(os.path.join(args.data_dir, 'test'), method='test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
    num_workers=8, pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set vis gpu
    device = torch.device('cuda')
    model = fusion_model().to(device)


    input1 = torch.randn(1, 3, 256, 256).to(device)
    input2 = torch.randn(1, 3, 256, 256).to(device)


    flops, params = profile(model, inputs=((input1,input2),))

    print(f"GFLOPs: {flops / 1e9}")
    print(f"Parameters: {params / 1e6}M")






