
from utils.regression_trainer import RegTrainer
import numpy as np
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default='/home/yyb/YYB/RGB-TT/ShanghaiTechRGBD',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/yyb/YYB/RGB-TT/exp',
                        help='directory to save models.')
    parser.add_argument('--dataset', default='ShanghaiTechRGBD',
                        help='Choose the dataset: RGBTCC or ShanghaiTechRGBD')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='default 256 for rgbtcc or 1024 for shanghai')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--resume', default='/home/yyb/YYB/RGB-TT/exp/0930-142416/299_ckpt.tar',
                        help='the path of resume training model')
    parser.add_argument('--device', default='0', help='assign device')


    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=500,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=50,
                        help='the epoch start to val')
    parser.add_argument('--save-all-best', type=bool, default=True,
                        help='whether to load opt state')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')
    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set vis gpu
    # torch.multiprocessing.set_start_method('spawn')
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()


