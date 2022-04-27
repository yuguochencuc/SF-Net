import os

# front-end parameter settings
fs = 48000
win_size = 960
fft_num = 960
win_shift = 480
is_chunk = True
chunk_length = 4*48000
feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x
is_as = False   # whether to conduct ablation study
is_pesq_criterion = False

#file_path = '/home/yuguochen/DNS-dataset-2022/DNS_100h_readspeech/'
#json_dir = '/home/yuguochen/DNS-dataset-2022/DNS_100h_readspeech/Json'
#data_type = 'dns'   # choices: dns_3000h, wsj0-si84-300h, vociebank
#file_path = '/home/yuguochen/vbdataset/VB_48k/'
#json_dir = '/home/yuguochen/CYCLEGAN-ATT-UNET/data/Json'
#data_type = 'vociebank'   # choices: dns_3000h, wsj0-si84-300h, vociebank

file_path = '/home/yuguochen/DNS-dataset-2022/DNS_dataset2022_600h/'
json_dir = '/home/yuguochen/DNS-dataset-2022/DNS_dataset2022_600h/Json'
data_type = 'dns'   # choices: dns_3000h, wsj0-si84-300h, vociebank


# network parameter settings
X= 6
R_sub = 4
is_gate = True

task_type = 'refine'
pretrained_model1_path = './LF_Net/BEST_MODEL/lfnet_dns300.pth.tar'
# server parameter settings
if is_as:
    loss_dir = './LOSS/full_Refine_dns600h.mat'
    best_path = './BEST_MODEL/full_Refine_dns600h/'
    checkpoint_path = './VB48k_CP/full_Refine_dns600h'
else:
    loss_dir = './LOSS/full_Refine_dns600h.mat'
    best_path = './BEST_MODEL/full_Refine_dns600h/'
    checkpoint_path = './VB48k_CP/full_Refine_dns600h'
batch_size = 8
epochs = 60
lr = 8e-4
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs('./BEST_MODEL/full_Refine_dns600h', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)

is_conti = False   # whether to load the checkpoint
is_cp = False    # whether to save the checkpoint
conti_path = ''  # the path of checkpoint
gpu_device = '6'