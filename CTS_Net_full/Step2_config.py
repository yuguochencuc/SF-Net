import os

win_size = 960
fft_num = 960
win_shift = 480
chunk_length = 3*48000

feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x

X = 6
R = 3

# network parameter setings
alpha = 0.1

# server parameter settings
file_path = '/home/yuguochen/vbdataset/VB_48k/'
json_dir = '/home/yuguochen/CYCLEGAN-ATT-UNET/data/Json'
loss_dir = './LOSS/stage2_vb_full.mat'
batch_size = 8
epochs = 50
lr = 1e-3
pretrained_model1_path = './BEST_MODEL/stage1_vb_full_pretrain.pth'
pretrained_model2_path = './BEST_MODEL/model2_second_update_best_model_epoch19.pth'
model_save_path = './MODEL'
model_best_path = './BEST_MODEL'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
