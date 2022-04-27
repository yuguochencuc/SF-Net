# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:26:14 2021

@author: Administrator
"""

import argparse
from stage1_net import Step1_net
import os
from Step1_train import *
from Step1_config import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(
    "Step1: coarsely suppress the noise"
)

parser.add_argument('--json_dir', type=str, default=json_dir,
                    help='The directory of the dataset feat,json format')
parser.add_argument('--loss_dir', type=str, default=loss_dir,
                    help='The directory to save tr loss and cv loss')
parser.add_argument('--batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--cv_batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--epochs', type=int, default=epochs,
                    help='The number of the training epoch')
parser.add_argument('--lr', type=float, default=lr,
                    help='Learning rate of the network')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--half_lr', type=int, default=1,
                    help='Whether to decay learning rate to half scale')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Whether to shuffle within each batch')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers to generate batch')
parser.add_argument('--l2', type=float, default=1e-7,
                    help='weight decay (L2 penalty)')
parser.add_argument('--best_path', default=best_path,
                    help='Location to save best cv model')
parser.add_argument('--print_freq', type=int, default=200,
                    help='The frequency of printing loss infomation')

# select GPU device
train_model = Step1_net()

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    # count the parameter number of the network
    model = train_model
    print(args)
    print('The number of trainable parameters of the first net is:%d' % (numParams(model)))
    main(args, model)