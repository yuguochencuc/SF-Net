import os

# front-end parameter settings
win_size = 960
fft_num = 960
win_shift = 480
chunk_length = 8*16000
feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x
is_conti = False
conti_path = './CP_dir/checkpoint_early_exit_16th.pth.tar'
is_pesq = False
is_cp= False


# server parameter settings
file_path = '/home/yuguochen/DNS-dataset-2022/DNS_dataset2022_600h/'
json_dir = '/home/yuguochen/DNS-dataset-2022/DNS_dataset2022_600h/Json'
data_type = 'dns'   # choices: dns_3000h, wsj0-si84-300h, vociebank

X= 6
R_sub = 4
is_gate = True

task_type = 'refine'

loss_dir = './LOSS/dsnet_dns600h.mat'
model_best_path = './BEST_MODEL/dsnet_dns600h.pth.tar'
check_point_path = './CP_dir/dsnet_dns600h'

loss_dir = './LOSS/dsnet_dns600h.mat'
batch_size = 8
epochs = 60
lr = 1e-3

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)