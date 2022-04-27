import torch
import torch.nn as nn
from torch.autograd import Variable
from ptflops import get_model_complexity_info
import torch.nn.functional as F
from Backup import numParams
from torch.nn.parameter import Parameter


class gcrn_net(nn.Module):
    def __init__(self, g_num, is_causal):
        super(gcrn_net, self).__init__()
        self.g_num = g_num
        self.en = Encoder()
        self.glstm1 = GLSTM(i_num=1024, g_num=g_num, is_causal=is_causal)
        self.glstm2 = GLSTM(i_num=1024, g_num=g_num, is_causal=is_causal)
        self.de1 = Decoder()
        self.de2 = Decoder()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        x, x_list = self.en(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, -1)
        x = self.glstm1(x)
        x = self.glstm2(x)
        x = x.view(batch_size, seq_len, 256, 4)
        x = x.permute(0, 2, 1, 3).contiguous()
        x_real = self.de1(x, x_list)
        x_imag = self.de2(x, x_list)
        del x_list
        return torch.stack((x_real, x_imag), dim=1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        en1 = nn.Sequential(
           Gate_Conv(2, 64, (1, 3), (1, 2), de_flag=0),
           nn.BatchNorm2d(64),
           nn.ELU())
        en2 = nn.Sequential(
           Gate_Conv(64, 64, (1, 3), (1, 2), de_flag=0),
           nn.BatchNorm2d(64),
           nn.ELU())
        en3 = nn.Sequential(
           Gate_Conv(64, 64, (1, 3), (1, 2), de_flag=0),
           nn.BatchNorm2d(64),
           nn.ELU())
        en4 = nn.Sequential(
           Gate_Conv(64, 64, (1, 3), (1, 2), de_flag=0),
           nn.BatchNorm2d(64),
           nn.ELU())
        en5 = nn.Sequential(
           Gate_Conv(64, 64, (1, 3), (1, 2), de_flag=0),
           nn.BatchNorm2d(64),
           nn.ELU())
        en6 = nn.Sequential(
           Gate_Conv_padding(64, 128, (1, 3), (1, 2), de_flag=0),
           nn.BatchNorm2d(128),
           nn.ELU())
        en7 = nn.Sequential(
           Gate_Conv_padding(128, 256, (1, 3), (1, 2), de_flag=0),
           nn.BatchNorm2d(256),
           nn.ELU())
        self.Module_list = nn.ModuleList([en1, en2, en3, en4, en5, en6, en7])

    def forward(self, x):
        x_list = []
        for i in range(len(self.Module_list)):
            x = self.Module_list[i](x)
            x_list.append(x)
        return x, x_list

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pad = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        de1 = nn.Sequential(
            Gate_Conv(2*256, 128, (1, 3), (1, 2), de_flag=1),
            Chomp_F(f=7),
            nn.BatchNorm2d(128),
            nn.ELU())
        de2 = nn.Sequential(
            Gate_Conv(2*128, 64, (1, 3), (1, 2), de_flag=1),
            Chomp_F(f=14),
            nn.BatchNorm2d(64),
            nn.ELU())
        de3 = nn.Sequential(
            Gate_Conv(2*64, 64, (1, 3), (1, 2), de_flag=1),
            nn.BatchNorm2d(64),
            nn.ELU())
        de4 = nn.Sequential(
            Gate_Conv(2*64, 64, (1, 3), (1, 2), de_flag=1),
            nn.BatchNorm2d(64),
            nn.ELU())
        de5 = nn.Sequential(
            Gate_Conv(2*64, 64, (1, 3), (1, 2), de_flag=1),
            nn.BatchNorm2d(64),
            nn.ELU())
        de6 = nn.Sequential(
            Gate_Conv(2*64, 64, (1, 3), (1, 2), de_flag=1),
            self.pad,
            nn.BatchNorm2d(64),
            nn.ELU())
        de7 = nn.Sequential(
            Gate_Conv(2*64, 1, (1, 3), (1, 2), de_flag=1),
            nn.BatchNorm2d(1),
            nn.ELU())
        self.Module_list = nn.ModuleList([de1, de2, de3, de4, de5,de6,de7])
        self.fc = nn.Linear(481, 481)


    def forward(self, x, x_list):
        for i in range(len(self.Module_list)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.Module_list[i](x)
        x = self.fc(x.squeeze(dim=1))
        return x

class GLSTM(nn.Module):
    def __init__(self, i_num, g_num, is_causal):
        super(GLSTM, self).__init__()
        self.K = g_num
        self.g_feat = i_num // self.K
        if is_causal is True:
            self.glstm_list = nn.ModuleList([nn.LSTM(self.g_feat, self.g_feat, batch_first=True) for i in range(self.K)])
        else:
            self.glstm_list = nn.ModuleList(
                [nn.LSTM(self.g_feat, self.g_feat//2, bidirectional=True, batch_first=True) for i in range(self.K)])

    def forward(self, x):
        batch_num, seq_len, feat_num = x.size()[0], x.size()[1], x.size()[2]
        x = x.reshape(batch_num, seq_len, self.K, self.g_feat)
        h = Variable(torch.zeros(batch_num, seq_len, self.K, self.g_feat)).to(x.device)
        for i in range(self.K):
            h[:, :, i, :], _ = self.glstm_list[i](x[:,:,i,:])
        h = h.permute(0, 1, 3, 2).contiguous()
        h = h.view(batch_num, seq_len, -1)
        return h

class Gate_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, de_flag):
        super(Gate_Conv, self).__init__()
        if de_flag == 0:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride)
            self.gate_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride),
                nn.Sigmoid())
        else:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride)
            self.gate_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride),
                nn.Sigmoid())

    def forward(self, x):
        return self.conv(x) * self.gate_conv(x)

class Gate_Conv_padding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, de_flag):
        super(Gate_Conv_padding, self).__init__()
        pad = (1, 1, 0, 0)
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
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride)
            self.gate_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride),
                nn.Sigmoid())

    def forward(self, x):
        return self.conv(x) * self.gate_conv(x)

class Chomp_F(nn.Module):
    def __init__(self, f):
        super(Chomp_F, self).__init__()
        self.f = f

    def forward(self, x):
        return x[:, :, :, 0:self.f]


if __name__ == '__main__':
    model1 = gcrn_net(g_num =2,  is_causal = True)

    print('The number of parameters of the model is:%.5d' % numParams(model1))
    x = torch.rand([4,2,101,481])
    y= model1(x)
    print(str(y.shape))
    macs, params = get_model_complexity_info(model1, (2,101, 481), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)