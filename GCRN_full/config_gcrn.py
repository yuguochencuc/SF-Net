import os

# front-end parameter settings
fs = 48000
win_size = 960
fft_num = 960
win_shift = 480
is_chunk = True
chunk_length = 3*48000
feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x
is_conti = False
conti_path = './CP_dir/checkpoint_early_exit_16th.pth.tar'

# server parameter settings
file_path = '/home/yuguochen/vbdataset/VB_48k/'
json_dir = '/home/yuguochen/CYCLEGAN-ATT-UNET/data/Json'
loss_dir = './LOSS/gcrn_vb_full_chomp.mat'
batch_size =8
epochs = 80
lr = 8e-4
model_best_path = './BEST_MODEL/gcrn_vb_full_chomp.pth.tar'
check_point_path = './CP_dir/gcrn_vb_full_chomp'
is_cp = False    # whether to save the checkpoint
is_pesq= False

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)

