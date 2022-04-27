import os

# front-end parameter settings
win_size = 960
fft_num = 960
win_shift = 480
chunk_length = 3*16000
feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x
is_conti = False
conti_path = './BEST_MODEL/MENet_0306_vb.pth.tar'
is_pesq = False
is_cp= False


X= 6
R_full = 4
R_sub = 4
is_gate = True


# server parameter settings
file_path = '/home/yuguochen/vbdataset/VB_48k/'
json_dir = '/home/yuguochen/CYCLEGAN-ATT-UNET/data/Json'
data_type = 'vociebank'   # choices: dns_3000h, wsj0-si84-300h, vociebank
loss_dir = './LOSS/MENet_0306_vb.mat'
batch_size = 16
epochs = 80
lr = 8e-4
model_best_path = './BEST_MODEL/MENet_0306_vb.pth.tar'
check_point_path = './CP_dir/MENet_0306_vb'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)