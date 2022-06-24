import os

# front-end parameter settings
fs = 48000
win_size = 960
fft_num = 960
win_shift = 480
is_chunk = True
chunk_length = 3*48000
feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x
is_as = False   # whether to conduct ablation study
is_pesq_criterion = False

#file_path = '/home/yuguochen/DNS-dataset-2022/DNS_100h_readspeech/'
#json_dir = '/home/yuguochen/DNS-dataset-2022/DNS_100h_readspeech/Json'
#data_type = 'dns'   # choices: dns_3000h, wsj0-si84-300h, vociebank
file_path = '/home/yuguochen/vbdataset/VB_48k/'
json_dir = '/home/yuguochen/CYCLEGAN-ATT-UNET/data/Json'
data_type = 'vociebank'   # choices: dns_3000h, wsj0-si84-300h, vociebank

#file_path = '/home/yuguochen/DNS-dataset-2022/DNS_dataset2022_600h/'
#json_dir = '/home/yuguochen/DNS-dataset-2022/DNS_dataset2022_600h/Json'
#data_type = 'dns'   # choices: dns_3000h, wsj0-si84-300h, vociebank


# network parameter settings
X= 6
R_sub = 4
is_gate = True

task_type = 'refine'
pretrained_model1_path = './LF-Net/BEST_MODEL/lfnet_0222_tcm1_inter.pth.tar'
# server parameter settings
if is_as:
    loss_dir = './LOSS/Refine_dns600h.mat'
    best_path1 = './BEST_MODEL/vb48k/lfnet_update.pth.tar'
    best_path2 = './BEST_MODEL/vb48k/mhfnet.pth.tar'
    checkpoint_path = './VB48k_CP/lmhf_net_test'
else:
    loss_dir = './LOSS/refine_high_gain.mat'
    best_path1 = './BEST_MODEL/vb48k/lfnet_update.pth.tar'
    best_path2 = './BEST_MODEL/vb48k/mhfnet.pth.tar'
    checkpoint_path = './VB48k_CP/lmhf_net_test'
batch_size = 16
epochs = 100
lr = 8e-4
os.makedirs(checkpoint_path, exist_ok=True)
is_conti = False   # whether to load the checkpoint
is_cp = True    # whether to save the checkpoint
conti_path = ' '  # the path of checkpoint
gpu_device = '5'