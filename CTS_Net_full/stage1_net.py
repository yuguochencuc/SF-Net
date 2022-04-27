# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:36:42 2021

@author: Administrator
"""


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from ptflops import get_model_complexity_info
from Backup import numParams
# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

# class Step1_net(nn.Module):
#     def __init__(self):
#         super(Step1_net, self).__init__()
#         self.en = Encoder()
#         self.de = Decoder()
#         self.lstm_module = LSTM_module()
#
#     def forward(self, x):
#         x = x.unsqueeze(dim=1)
#         x, x_list = self.en(x)
#         batch_num, _, seq_len, _ = x.shape
#         x = x.permute(0, 2, 1, 3).contiguous()
#         x = x.view(batch_num, seq_len, -1)
#         x = self.lstm_module(x)
#         x = x.view(batch_num, seq_len, 64, 4)
#         x = x.permute(0, 2, 1, 3).contiguous()
#         x = self.de(x, x_list)
#         del x_list
#         return x

class Step1_net(nn.Module):
    def __init__(self):
        super(Step1_net, self).__init__()
        self.en = Encoder()
        self.de = Decoder()
        self.tcm1 = Tcm_list(X=6)
        self.tcm2 = Tcm_list(X=6)
        self.tcm3 = Tcm_list(X=6)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x, x_list = self.en(x)
        batch_num, _, seq_len, _ = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_num, -1, seq_len)
        x_acc = Variable(torch.zeros(x.size()), requires_grad=True).to(x.device)
        x = self.tcm1(x)
        x_acc = x_acc + x
        x = self.tcm2(x)
        x_acc = x_acc + x
        x = self.tcm3(x)
        x_acc = x_acc + x
        x = x_acc
        del x_acc
        x = x.view(batch_num, 64, 4, seq_len)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.de(x, x_list)
        del x_list
        return x

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls()
        model.load_state_dict(package['state_dict'])
        return model
        
    @classmethod    
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None, lr=None):
        package = {
            'en': model.en,
            'de': model.de,
            'tcm1': model.tcm1,
            'tcm2': model.tcm2,
            'tcm3': model.tcm3,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        if lr is not None:
            package['lr'] = lr
        return package    


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        en1 = nn.Sequential(
            Gate_Conv(1, 64, kernel_size=(2, 5), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en2 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en3 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en4 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en5 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en6 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(1, 1, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en7 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(1, 1, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        self.en = nn.ModuleList([en1, en2, en3, en4, en5, en6, en7])

    def forward(self, x):
        x_list = []
        for i in range(len(self.en)):
            x = self.en[i](x)
            x_list.append(x)
        return x, x_list


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        de1 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 1), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de2 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 2), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de3 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de4 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de5 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de6 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de7 = nn.Sequential(
            Gate_Conv(64*2, 1, kernel_size=(2, 5), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(1, affine=True),
            nn.PReLU(1)
        )
        self.de8 = nn.Sequential(
            nn.Linear(481, 481),
            nn.Softplus())
        self.de = nn.ModuleList([de1, de2, de3, de4, de5, de6, de7])

    def forward(self, x, x_list):
        for i in range(len(x_list)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.de[i](x)
        x = self.de8(x.squeeze(dim=1))
        return x

class LSTM_module(nn.Module):
    def __init__(self):
        super(LSTM_module, self).__init__()
        self.lstm = nn.LSTM(256, 256, batch_first=True, num_layers=2)
        self.fc = nn.Linear(256, 256)
    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.fc(h)
        x = x + h
        return x

class Gate_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, de_flag, pad=(0, 0, 0, 0), chomp=1):
        super(Gate_Conv, self).__init__()
        if de_flag == 0:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(pad, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride))
            self.gate_conv = nn.Sequential(
                nn.ConstantPad2d(pad, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride),
                nn.Sigmoid())
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride),
                Chomp_T(chomp))
            self.gate_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels= out_channels,
                                                kernel_size=kernel_size, stride=stride),
                Chomp_T(chomp),
                nn.Sigmoid())
    def forward(self, x):
        return self.conv(x) * self.gate_conv(x)


class Tcm_list(nn.Module):
    def __init__(self, X):
        super(Tcm_list, self).__init__()
        self.X = X
        self.tcm_list = nn.ModuleList([Glu(2 ** i) for i in range(self.X)])
    def forward(self, x):
        for i in range(self.X):
            x = self.tcm_list[i](x)
        return x

class Glu(nn.Module):
    def __init__(self, dilation):
        super(Glu, self).__init__()
        self.in_conv = nn.Conv1d(256, 64, kernel_size=1, bias=False)
        self.left_conv = nn.Sequential(
            nn.PReLU(64),
            nn.InstanceNorm1d(64, affine=True),
            ShareSepConv(2*dilation-1),
            nn.ConstantPad1d((4 * dilation, 0), value=0.),
            nn.Conv1d(64, 64, kernel_size=5, dilation=dilation, bias=False)
        )
        self.right_conv = nn.Sequential(
            nn.PReLU(64),
            nn.InstanceNorm1d(64, affine=True),
            ShareSepConv(2 * dilation - 1),
            nn.ConstantPad1d((4 * dilation, 0), value=0.),
            nn.Conv1d(64, 64, kernel_size=5, dilation=dilation, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(64),
            nn.InstanceNorm1d(64, affine=True),
            nn.Conv1d(64, 256, kernel_size=1, bias=False)
        )
    def forward(self, x):
        resi = x
        x = self.in_conv(x)
        x = self.left_conv(x) * self.right_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        self.pad = nn.ConstantPad1d((kernel_size - 1, 0), value=0.)
        weight_tensor = torch.zeros(1, 1, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size).contiguous()
        x = self.pad(x)
        x = F.conv1d(x, expand_weight, None, stride=1, dilation=1, groups=inc)
        return x

class Chomp_T(nn.Module):
    def __init__(self, t):
        super(Chomp_T, self).__init__()
        self.t = t
    def forward(self, x):
        return x[:, :, 0:-self.t, :]

if __name__ == '__main__':
    model1 = Step1_net()

    print('The number of parameters of the model is:%.5d' % numParams(model1))
    x = torch.rand([4,101,481])
    y= model1(x)
    print(str(y.shape))
    macs, params = get_model_complexity_info(model1, (101, 481), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)