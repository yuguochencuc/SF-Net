import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from ptflops import get_model_complexity_info
import torch.nn.functional as F
from Backup import numParams
from torch.nn.parameter import Parameter
from utils import NormSwitch


# class complex_denoise_net(nn.Module):
#     def __init__(self,):
#         super(complex_denoise_net, self).__init__()
#         self



class DB_denoise_net_tcm1(nn.Module):
    def __init__(self, X, R, is_gate=True):
        super(DB_denoise_net_tcm1, self).__init__()
        self.X, self.R = X, R
        self.is_gate = is_gate
        self.en_mag = Encoder(ci=1, is_gate=self.is_gate)
        self.en_ri = Encoder(ci=2, is_gate=self.is_gate)
        self.de_mask = Decoder_mask( is_gate=self.is_gate)
        self.de_r = Decoder(is_gate=self.is_gate)
        self.de_i = Decoder(is_gate=self.is_gate)
        self.tcm_list = nn.ModuleList([Tcm_list(X=self.X) for _ in range(self.R)])
        #self.tcm_list_ri = nn.ModuleList([Tcm_list(X=self.X) for _ in range(self.R)])
        self.out = nn.ReLU(inplace=True)
        self.inter = interaction(64)
        self.aham = AHAM()

    def forward(self, x):
        #input = input.unsqueeze(dim=1)
        batch_size, _, seq_len, _ = x.shape
        x_r_input, x_i_input = x[:,0,:,:], x[:,1,:,:]
        x_mag_ori, x_phase_ori = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :]) #BTF BTF
        x_mag = x_mag_ori.unsqueeze(dim = 1) #B1TF
        x_mag, x_mag_list = self.en_mag(x_mag)
        x_ri, x_ri_list = self.en_ri(x)
        x_mag_inter = self.inter(x_mag, x_ri)
        x_ri_inter = self.inter(x_ri, x_mag)

        batch_num, _, seq_len, _ = x_mag.shape
        x_mag_en = x_mag_inter.permute(0, 1, 3, 2).contiguous()
        x_mag_en = x_mag_en.view(batch_num, -1, seq_len)
        x_mag_en_acc = Variable(torch.zeros(x_mag_en.size()), requires_grad=True).to(x_mag_en.device)
        mag_tcm_output= []
        for i in range(len(self.tcm_list)):
            x_mag_en= self.tcm_list[i](x_mag_en)
            x_mag_en_acc = x_mag_en_acc + x_mag_en
            mag_tcm_output.append(x_mag_en_acc)
        x_mag_tcm = x_mag_en_acc
        del x_mag_en_acc
        x_mag_aham = self.aham(mag_tcm_output)
        x_mag = x_mag_aham.view(batch_num, 64, 4, seq_len)
        x_mag = x_mag.permute(0, 1, 3, 2).contiguous()
        x_mag_mask = self.de_mask(x_mag, x_mag_list)
        #out_mag = input * x_mag_mask
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        x_ri_en = x_ri_inter.permute(0, 1, 3, 2).contiguous()
        x_ri_en = x_ri_en.view(batch_num, -1, seq_len)
        x_ri_tcm_acc = Variable(torch.zeros(x_ri_en.size()), requires_grad=True).to(x_ri_en.device)
        ri_tcm_output= []
        for i in range(len(self.tcm_list)):
            x_ri_en = self.tcm_list[i](x_ri_en)
            x_ri_tcm_acc = x_ri_tcm_acc + x_ri_en
            ri_tcm_output.append(x_ri_tcm_acc)
        x_ri_tcm = x_ri_tcm_acc
        del x_ri_tcm_acc
        x_ri_aham = self.aham(ri_tcm_output)
        x_ri = x_ri_aham.view(batch_num, 64, 4, seq_len)
        x_ri = x_ri.permute(0, 1, 3, 2).contiguous()
        x_r, x_i = self.de_r(x_ri, x_ri_list), self.de_i(x_ri, x_ri_list)
        x_r = x_r.squeeze(dim = 1)
        x_i = x_i.squeeze(dim=1)
        x_mag_out = x_mag_mask * x_mag_ori

        x_r_out, x_i_out = (x_mag_out * torch.cos(x_phase_ori) + x_r), (x_mag_out * torch.sin(x_phase_ori)+ x_i)

        x_com_out = torch.stack((x_r_out,x_i_out),dim=1)

        return x_com_out

class DB_denoise_net(nn.Module):
    def __init__(self, X, R, is_gate=True):
        super(DB_denoise_net, self).__init__()
        self.X, self.R = X, R
        self.is_gate = is_gate
        self.en_mag = Encoder(ci=1, is_gate=self.is_gate)
        self.en_ri = Encoder(ci=2, is_gate=self.is_gate)
        self.de_mask = Decoder_mask( is_gate=self.is_gate)
        self.de_r = Decoder(is_gate=self.is_gate)
        self.de_i = Decoder(is_gate=self.is_gate)
        self.tcm_list_mag = nn.ModuleList([Tcm_list(X=self.X) for _ in range(self.R)])
        self.tcm_list_ri = nn.ModuleList([Tcm_list(X=self.X) for _ in range(self.R)])
        self.out = nn.ReLU(inplace=True)
        self.aham = AHAM()

    def forward(self, x):
        #input = input.unsqueeze(dim=1)
        batch_size, _, seq_len, _ = x.shape
        x_r_input, x_i_input = x[:,0,:,:], x[:,1,:,:]
        x_mag_ori, x_phase_ori = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :]) #BTF BTF
        x_mag = x_mag_ori.unsqueeze(dim = 1) #B1TF
        x_mag, x_mag_list = self.en_mag(x_mag)
        batch_num, _, seq_len, _ = x_mag.shape
        x_mag_en = x_mag.permute(0, 1, 3, 2).contiguous()
        x_mag_en = x_mag_en.view(batch_num, -1, seq_len)
        x_mag_en_acc = Variable(torch.zeros(x_mag_en.size()), requires_grad=True).to(x_mag_en.device)
        mag_tcm_output= []
        for i in range(len(self.tcm_list_mag)):
            x_mag_tcm= self.tcm_list_mag[i](x_mag_en)
            x_mag_en_acc = x_mag_en_acc + x_mag_tcm
            mag_tcm_output.append(x_mag_en_acc)
        x_mag_tcm = x_mag_en_acc
        del x_mag_en_acc
        x_mag_aham = self.aham(mag_tcm_output)
        x_mag = x_mag_aham.view(batch_num, 64, 4, seq_len)
        x_mag = x_mag.permute(0, 1, 3, 2).contiguous()
        x_mag_mask = self.de_mask(x_mag, x_mag_list)
        #out_mag = input * x_mag_mask
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        x_ri, x_ri_list = self.en_ri(x)
        x_ri_en = x_ri.permute(0, 1, 3, 2).contiguous()
        x_ri_en = x_ri_en.view(batch_num, -1, seq_len)
        x_ri_tcm_acc = Variable(torch.zeros(x_ri_en.size()), requires_grad=True).to(x_ri_en.device)
        ri_tcm_output= []
        for i in range(len(self.tcm_list_ri)):
            x_ri_tcm = self.tcm_list_ri[i](x_ri_en)
            x_ri_tcm_acc = x_ri_tcm_acc + x_ri_tcm
            ri_tcm_output.append(x_ri_tcm_acc)
        x_ri_tcm = x_ri_tcm_acc
        del x_ri_tcm_acc
        x_ri_aham = self.aham(ri_tcm_output)
        x_ri = x_ri_aham.view(batch_num, 64, 4, seq_len)
        x_ri = x_ri.permute(0, 1, 3, 2).contiguous()
        x_r, x_i = self.de_r(x_ri, x_ri_list), self.de_i(x_ri, x_ri_list)
        x_r = x_r.squeeze(dim = 1)
        x_i = x_i.squeeze(dim=1)
        x_mag_out = x_mag_mask * x_mag_ori

        x_r_out, x_i_out = (x_mag_out * torch.cos(x_phase_ori) + x_r), (x_mag_out * torch.sin(x_phase_ori)+ x_i)

        x_com_out = torch.stack((x_r_out,x_i_out),dim=1)

        return x_com_out


class Encoder(nn.Module):
    def __init__(self, ci, is_gate):
        super(Encoder, self).__init__()
        self.ci, self.is_gate = ci, is_gate
        en1 = Conv_block(self.ci, 64, (2, 5), self.is_gate)
        en2 = Conv_block(64, 64, (2, 3), self.is_gate)
        en3 = Conv_block(64, 64, (2, 3), self.is_gate)
        en4 = Conv_block(64, 64, (2, 3), self.is_gate)
        en5 = Conv_block(64, 64, (2, 3), self.is_gate)
        self.en = nn.ModuleList([en1, en2, en3, en4, en5])

    def forward(self, x):
        x_list = []
        for i in range(len(self.en)):
            x = self.en[i](x)
            x_list.append(x)
        return x, x_list

class Decoder_mask(nn.Module):
    def __init__(self,  is_gate):
        super(Decoder_mask, self).__init__()
        self.is_gate = is_gate
        de1 = Deconv_block(64, 64, (2, 3), is_gate=self.is_gate)
        de2 = Deconv_block(64, 64, (2, 3), is_gate=self.is_gate)
        de3 = Deconv_block(64, 64, (2, 3), is_gate=self.is_gate)
        de4 = Deconv_block(64, 64, (2, 3), is_gate=self.is_gate)
        de5 = Deconv_block(64, 1, (2, 5), is_gate=self.is_gate)
        self.de = nn.ModuleList([de1, de2, de3, de4, de5])
        self.mask1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1)),
            nn.Tanh()
        )
        self.maskconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.mask_final = nn.Sigmoid()

    def forward(self, x, x_list):
        for i in range(len(x_list)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.de[i](x)
        out = self.mask1(x) * self.mask2(x)
        out = self.mask_final(self.maskconv(out))
        return out

class Decoder(nn.Module):
    def __init__(self, is_gate):
        super(Decoder, self).__init__()
        self.is_gate = is_gate
        de1 = Deconv_block(64, 64, (2, 3), is_gate=self.is_gate)
        de2 = Deconv_block(64, 64, (2, 3), is_gate=self.is_gate)
        de3 = Deconv_block(64, 64, (2, 3), is_gate=self.is_gate)
        de4 = Deconv_block(64, 64, (2, 3), is_gate=self.is_gate)
        de5 = Deconv_block(64, 1, (2, 5), is_gate=self.is_gate)
        self.de = nn.ModuleList([de1, de2, de3, de4, de5])
        self.de6 = nn.Linear(161, 161)

    def forward(self, x, x_list):
        for i in range(len(x_list)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.de[i](x)
        x = self.de6(x.squeeze(dim=1))
        return x


class Conv_block(nn.Module):
    def __init__(self, ci, co, k, is_gate):
        self.ci, self.co, self.k, self.is_gate = ci, co, k, is_gate
        super(Conv_block, self).__init__()
        #self.norm = NormSwitch('cLN', '2D', self.co)
        conv_list = []
        if self.is_gate is True:
            conv_list.append(Gate_Conv(self.ci, self.co, self.k, stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)))
        else:
            conv_list.append(nn.ConstantPad2d((0, 0, 1, 0), value=0.))
            conv_list.append(nn.Conv2d(self.ci, self.co, kernel_size=self.k, stride=(1, 2)))
        conv_list.append(NormSwitch('cLN', '2D', self.co))
        conv_list.append(nn.PReLU(self.co))
        self.convs = nn.Sequential(*conv_list)

    def forward(self, x):
        x = self.convs(x)
        return x


class Deconv_block(nn.Module):
    def __init__(self, ci, co, k, is_gate):
        self.ci, self.co, self.k, self.is_gate = ci, co, k, is_gate
        super(Deconv_block, self).__init__()
        #self.norm = NormSwitch('cLN', '2D', self.co)
        conv_list = []
        if self.is_gate is True:
            conv_list.append(Gate_Conv(self.ci * 2, self.co, kernel_size=k, stride=(1, 2), de_flag=1, chomp=1))
        else:
            conv_list.append(nn.ConvTranspose2d(self.ci * 2, self.co, kernel_size=k, stride=(1, 2)))
            conv_list.append(Chomp_T(1))
        conv_list.append(NormSwitch('cLN', '2D', self.co))
        conv_list.append(nn.PReLU(self.co))
        self.convs = nn.Sequential(*conv_list)

    def forward(self, x):
        x = self.convs(x)
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
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride),
                nn.Sigmoid())
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride),
                Chomp_T(chomp))
            self.gate_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride),
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
            NormSwitch('cLN','1D',64),
            ShareSepConv(2 * dilation - 1),
            nn.ConstantPad1d((4 * dilation, 0), value=0.),
            nn.Conv1d(64, 64, kernel_size=5, dilation=dilation, bias=False)
        )
        self.right_conv = nn.Sequential(
            nn.PReLU(64),
            NormSwitch('cLN','1D',64),
            ShareSepConv(2 * dilation - 1),
            nn.ConstantPad1d((4 * dilation, 0), value=0.),
            nn.Conv1d(64, 64, kernel_size=5, dilation=dilation, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(64),
            NormSwitch('cLN','1D',64),
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
        weight_tensor[0, 0, (kernel_size - 1) // 2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size).contiguous()
        x = self.pad(x)
        x = F.conv1d(x, expand_weight, None, stride=1, dilation=1, groups=inc)
        return x

class AHAM(nn.Module):  # aham merge
    def __init__(self,  input_channel=256, kernel_size=1, bias=True, act=nn.ReLU(True)):
        super(AHAM, self).__init__()

        self.k3 = Parameter(torch.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        #self.conv1=nn.Conv1d(input_channel, 1, kernel_size, stride=1, bias=bias)

    def merge(self, x, y):
        batch, channel, seq_len, frequency = x.size()
        input_x = x  # B*C*T*K
        # input_y = y #N*1*1*K*1
        y = self.softmax(y)
        context = torch.matmul(input_x, y)  # N*C*H*W*1
        context = context.view(batch, channel, seq_len)  # N*C*H*W

        return context

    def forward(self, input_list): #X:B(Cf)TG Y:BTG1
        batch, channel_frequency, frames= input_list[-1].size()
        x_list = []
        y_list = []
        for i in range(len(input_list)):
            #batch_num, channel, seq_len, frequency = input_list[i].shape
            input_list[i] = input_list[i].permute(0, 2, 1).contiguous()
            input = self.avg_pool(input_list[i])
            y = input
            x = input_list[i].unsqueeze(-1)
            y= y.unsqueeze(-2)
            x_list.append(x)
            y_list.append(y)

        x_merge = torch.cat((x_list[0],x_list[1], x_list[2], x_list[3]), dim=-1)

        y_merge = torch.cat((y_list[0],y_list[1], y_list[2], y_list[3]), dim=-2)
        y_softmax = self.softmax(y_merge)
        aham= torch.matmul(x_merge, y_softmax)
        aham= aham.view(batch, channel_frequency, frames)
        input_list[-1] = input_list[-1].permute(0, 2, 1).contiguous()
        aham_output = input_list[-1] + aham
        return aham_output

class interaction(nn.Module):
    def __init__(self, input_size ):
        super(interaction,self).__init__()
        self.inter = nn.Sequential(
            nn.Conv2d(2 * input_size, input_size, kernel_size=(1,1)),
            NormSwitch('cLN','2D',64),
            nn.Sigmoid()
        )
        #self.input_1d = nn.Conv1d(2 * input_size, input_size, kernel_size=1),
        #self.norm = nn.InstanceNorm1d(input_size, affine=True),
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        input_merge = torch.cat((input1,input2),dim=1)
        output_mask = self.inter(input_merge)
        output = input1 + input2 * output_mask

        return  output

class interaction_ori(nn.Module):
    def __init__(self, input_size ):
        super(interaction_ori,self).__init__()
        self.input_conv1 = nn.Sequential(
            nn.PReLU(input_size),
            nn.InstanceNorm1d(input_size, affine=True),
            nn.Conv1d(input_size, 64, kernel_size=1, bias=False)
        )
        self.input_conv2 = nn.Sequential(
            nn.PReLU(256),
            nn.InstanceNorm1d(256, affine=True),
            nn.Conv1d(256, 64, kernel_size=1, bias=False)
        )
        self.inter = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 1),
            nn.InstanceNorm1d(256, affine=True),
            nn.Sigmoid()
        )
        #self.input_1d = nn.Conv1d(2 * input_size, input_size, kernel_size=1),
        #self.norm = nn.InstanceNorm1d(input_size, affine=True),
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        input1_conv = self.input_conv1(input1)
        input2_conv = self.input_conv2(input2)
        input_merge = torch.cat((input1_conv,input2_conv),dim=1)
        output_mask = self.inter(input_merge)
        output = input1 + input2 * output_mask

        return  output


class Chomp_T(nn.Module):
    def __init__(self, t):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, 0:-self.t, :]


if __name__ == '__main__':
    model1 = DB_denoise_net_tcm1(X=6 ,R=4, is_gate= True)

    print('The number of parameters of the model is:%.5d' % numParams(model1))
    x = torch.rand([4,2,101,161])
    y= model1(x)
    print(str(y.shape))
    macs, params = get_model_complexity_info(model1, (2,101, 161), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

# if __name__ == '__main__':
#     x = torch.rand([4,500,101])
#     x1 = torch.rand([4, 500, 101])
#     x = [x,x1,x1,x,x1,x]
#     aham= AHAM()
#     y= aham(x)
#     print(str(y.shape))
