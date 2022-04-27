# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:44:35 2021

@author: Administrator
"""

import torch
import argparse
import librosa
import os
import numpy as np
import json
import scipy
from Backup import *
import pickle
from Step1_config import *
from stage1_net import Step1_net
import soundfile as sf
from istft import ISTFT

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def enhance(args):
    model1 = Step1_net()
    model1.load_state_dict(torch.load(args.Step1_model_path))
    model1.cuda().eval()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_file_path
        seen_flag = 1
        # noise_type = args.noise_type
        # snr = args.snr
        file_list = os.listdir(mix_file_path)
        istft = ISTFT(filter_length=960, hop_length=480, window='hanning')
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 960 + 960) / 480 + 1))
            fake_wav_len = (frame_num - 1) * 480 + 960 - 960
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=960, hop_length=480, win_length=960,
                                window=torch.hann_window(960)).permute(0, 3, 2, 1)
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :]).cuda()
            #print(str(noisy_phase.shape))
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5

            # the first step
            esti_x = model1(feat_x_mag.cuda())
            esti_x = esti_x ** 2
            #print(str(esti_x.shape))
            esti_com = torch.stack((esti_x*torch.cos(noisy_phase), esti_x*torch.sin(noisy_phase)), dim=1)

            esti_com = esti_com.cpu()
            esti_utt = istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            os.makedirs(os.path.join(esti_file_path), exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % ((cnt + 1)))
            cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='/home/yuguochen/vbdataset/VB_48k/noisy_testset_wav/')
    parser.add_argument('--esti_file_path', type=str, default='./enhanced_wavs/VB_48k/stage1_vb_full')
    # parser.add_argument('--noise_type', type=str, default='factory1')  # babble   cafe  factory1
    # parser.add_argument('--seen_flag', type=int, default=0)    # 1   0
    # parser.add_argument('--snr', type=int, default=5)          # -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=48000,
                        help='The sampling rate of speech')
    parser.add_argument('--Step1_model_path', type=str,
                        default='./BEST_MODEL/stage1_vb_full_pretrain.pth',
                        help='The place to save best model')
    args = parser.parse_args()
    print(args)
    enhance(args=args)