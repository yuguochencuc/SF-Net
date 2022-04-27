import argparse
from stage1_net import Step1_net
from stage2_net import Step2_net
import os
from Step2_train import *
from Step2_config import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(
    "Step2: decorate the speeech componnets in the complex domain"
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
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--half_lr', type=int, default=1,
                    help='Whether to decay learning rate to half scale')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Whether to shuffle within each batch')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers to generate batch')
parser.add_argument('--l2', type=float, default=1e-7,
                    help='weight decay (L2 penalty)')
parser.add_argument('--pretrained_model1_path', default=pretrained_model1_path)
parser.add_argument('--pretrained_model2_path', default=pretrained_model2_path)
parser.add_argument('--print_freq', type=int, default=200,
                    help='The frequency of printing loss infomation')
parser.add_argument('--model_save_path', type=str, default=model_save_path,
                    help='')
parser.add_argument('--model_best_path', type=str, default=model_best_path,
                    help='')

train_model1 = Step1_net()
train_model2 = Step2_net(X=X, R=R)

if __name__ == '__main__':
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True
    args = parser.parse_args()
    model = [train_model1, train_model2]
    print(args)
    main(args, model)