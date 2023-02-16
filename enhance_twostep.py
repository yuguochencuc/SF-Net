import torch
import argparse
import librosa
import os
import numpy as np
from istft import ISTFT
from LF_Net.denoisenet_low import DB_denoise_net_tcm1
from denoisenet_high import denoise_highband_net
import soundfile as sf


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def enhance(args):
    model1 = DB_denoise_net_tcm1(X=6, R=4, is_gate = True)
    model2 = denoise_highband_net(X=6, R_sub=4, is_gate = True)

    model1.load_state_dict(torch.load(args.Step1_model_path))
    model1.cuda().eval()
    model2.load_state_dict(torch.load(args.Step2_model_path))
    model2.cuda().eval()

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
            #low_frequency process
            s1_com_out_low = feat_x[:,:,:,0:161].cuda()
            s1_com_out_low_phase = torch.atan2(s1_com_out_low[:, -1, :, :], s1_com_out_low[:, 0, :, :])
            s1_com_out_low_mag = (torch.norm(s1_com_out_low, dim=1)) ** 0.5
            
            feat_x_low = torch.stack((s1_com_out_low_mag * torch.cos(s1_com_out_low_phase), s1_com_out_low_mag * torch.sin(s1_com_out_low_phase)), dim=1)
            
            esti_x_low = model1(feat_x_low)
            
            feat_x = torch.stack((feat_x_mag * torch.cos(noisy_phase), feat_x_mag * torch.sin(noisy_phase)), dim=1)
            feat_x_full = feat_x.cuda()
            feat_x_full[:,:,:,0:161] = esti_x_low
            #print(str(feat_x_full.shape))

            s4_out = model2(feat_x_full)
            s4_out_mag, s4_out_phase =  torch.norm(s4_out, dim=1), torch.atan2(s4_out[:, -1, :, :], s4_out[:, 0, :, :]) 
            #s4_out_mag_low = s4_out_mag[:,:,:,0:161]
            s4_out_mag = s4_out_mag ** 2
            esti_com = torch.stack((s4_out_mag*torch.cos(s4_out_phase), s4_out_mag*torch.sin(s4_out_phase)), dim=1)
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
    parser.add_argument('--mix_file_path', type=str, default='/home/yuguochen/interspeech2022_fullband/enhanced_wavs/test_1107')
    parser.add_argument('--esti_file_path', type=str, default='./enhanced_wavs/test_1107_enhanced')
    parser.add_argument('--snr', type=list, default=[-5, 0, 5, 10, 15, 20])     #  -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=48000,
                        help='The sampling rate of speech')
    parser.add_argument('--Step1_model_path', type=str, default='./BEST_MODEL/vb48k_0222/model1_second_update_best_model_epoch46.pth', help='The place to save best model')

    #parser.add_argument('--Step1_model_path', type=str,
                        #default='./LF_Net/BEST_MODEL/lfnet_dns300.pth.tar',default='./BEST_MODEL/full_Refine_dns600h/model1_second_update_best_model_epoch33.pth',help='The place to save best model')                        
    #parser.add_argument('--Step2_model_path', type=str, default='./BEST_MODEL/full_Refine_dns600h/model2_second_update_best_model_epoch33.pth', help='The place to save best model')
    parser.add_argument('--Step2_model_path', type=str, default='./BEST_MODEL/vb48k_0222/model2_second_update_best_model_epoch46.pth', help='The place to save best model')
                        

    args = parser.parse_args()
    print(args)
    enhance(args=args)