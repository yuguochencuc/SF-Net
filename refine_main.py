import argparse
from data_dns import TrainDataset, CvDataset, TrainDataLoader, CvDataLoader
from LF_Net.denoisenet_low import DB_denoise_net_tcm1
from denoisenet_high import denoise_highband_net
from config_refine import *
from solver_refine import Solver
from Backup import numParams, set_requires_grad
import torch
import torch.nn as nn
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
import warnings
warnings.filterwarnings('ignore')

# fix random seed
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)

parser = argparse.ArgumentParser(
    "Glance and Focus Network for monaural speech enhancement"
)

parser.add_argument('--json_dir', type=str, default=json_dir,
                    help='The directory of the dataset feat,json format')
parser.add_argument('--loss_dir', type=str, default=loss_dir,
                    help='The directory to save tr loss and cv loss')
parser.add_argument('--batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--cv_batch_size', type=int, default=batch_size,
                    help='The number of the batch size')
parser.add_argument('--epochs', type=int, default=epochs,
                    help='The number of the training epoch')
parser.add_argument('--lr', type=float, default=lr,
                    help='Learning rate of the network')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--half_lr', type=int, default=1,
                    help='Whether to decay learning rate to half scale')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Whether to shuffle within each batch')
parser.add_argument('--pre1_path', type=str, default=pretrained_model1_path)
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers to generate batch')
parser.add_argument('--l2', type=float, default=1e-7,
                    help='weight decay (L2 penalty)')
parser.add_argument('--best_path', default=best_path,
                    help='Location to save best cv model')
parser.add_argument('--cp_path', type=str, default=checkpoint_path)
parser.add_argument('--is_conti', type=bool, default=is_conti)
parser.add_argument('--is_cp', type=bool, default=is_cp)
parser.add_argument('--conti_path', type=str, default=conti_path)
parser.add_argument('--print_freq', type=int, default=1000,
                    help='The frequency of printing loss infomation')

# select model
step1_model = DB_denoise_net_tcm1(X=6, R=4, is_gate= True)
step2_model = denoise_highband_net(X=X, R_sub=R_sub, is_gate=is_gate)


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    model1 = step1_model
    model2 = step2_model
    model1.cuda()
    model2.cuda()
    model1.load_state_dict(torch.load(args.pre1_path))
    print(args)
    # count the parameter number of the network
    print('The number of trainable parameters of the first net is:%d' % (numParams(model2)))

    tr_dataset = TrainDataset(json_dir=args.json_dir,
                              batch_size=args.batch_size)
    cv_dataset = CvDataset(json_dir=args.json_dir,
                           batch_size=args.cv_batch_size)
    tr_loader = TrainDataLoader(data_set=tr_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                pin_memory=True)
    cv_loader = CvDataLoader(data_set=cv_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    optimizer = torch.optim.Adam([{'params': model1.parameters(), 'lr': 1e-4},
                                  {'params': model2.parameters()}],
                                 lr=args.lr,
                                 weight_decay=args.l2
                                 )
    solver = Solver(data, [model1, model2], optimizer, args)
    solver.train()