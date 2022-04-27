# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:32:13 2021

@author: Administrator
"""

import os

win_size = 960
fft_num = 960
win_shift = 480
chunk_length = 3*48000

feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x

X = 6
R = 3

# server configuration
file_path = '/home/yuguochen/vbdataset/VB_48k/'
json_dir = '/home/yuguochen/CYCLEGAN-ATT-UNET/data/Json'
loss_dir = './LOSS/stage1_vb_full_pretrain.mat'
batch_size = 6
epochs = 60
lr = 1e-3
best_path = './BEST_MODEL/stage1_vb_full_pretrain.pth'
checkpoint_path = './MODEL'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)