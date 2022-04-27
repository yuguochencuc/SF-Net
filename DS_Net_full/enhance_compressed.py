import torch
import argparse
import librosa
import os
import numpy as np
from istft import ISTFT
from denoisenet_full import DB_denoise_net_tcm1, CP_denoise_net_tcm1, ME_denoise_net_tcm1
import soundfile as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def enhance(args):
    model = ME_denoise_net_tcm1(X=6, R=4, is_gate = True)
    checkpoint = torch.load(args.Model_path)
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_file_path        
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
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :])
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5
            feat_x = torch.stack((feat_x_mag * torch.cos(noisy_phase), feat_x_mag * torch.sin(noisy_phase)), dim=1)
            esti_x = model(feat_x.cuda())
            esti_mag, esti_phase = torch.norm(esti_x, dim=1), torch.atan2(esti_x[:, -1, :, :],
                                                                             esti_x[:, 0, :, :])
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            esti_com = esti_com.cpu()
            esti_utt = istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            os.makedirs(os.path.join(esti_file_path), exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='/home/yuguochen/vbdataset/VB_48k/noisy_testset_wav/')
    parser.add_argument('--esti_file_path', type=str, default='./estimated_audio/menet_vb_48k_best')
    parser.add_argument('--snr', type=list, default=[-5, 0, 5, 10, 15, 20])     #  -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=48000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='./BEST_MODEL/MENet_0306_vb.pth.tar',
                        help='The place to save best model')
    args = parser.parse_args()
    print(args)
    enhance(args=args)