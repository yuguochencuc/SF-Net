import os

# front-end parameter settings
win_size = 320
fft_num = 320
win_shift = 160
chunk_length = 8*16000
feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x
is_conti = False
conti_path = './CP_dir/checkpoint_early_exit_16th.pth.tar'
is_pesq = False
is_cp= False


X= 6
R_full = 4
R_sub = 4
is_gate = True

# server parameter settings
json_dir = '/home/yuguochen/DNS_dataset_300h/Json'
file_path = '/home/yuguochen/DNS_dataset_300h/'
loss_dir = './LOSS/lfnet_dns300.mat'
batch_size = 16
epochs = 80
lr = 1e-3
model_best_path = './BEST_MODEL/lfnet_dns300.pth.tar'
check_point_path = './CP_dir/lfnet_dns300'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)