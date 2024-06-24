import torch
import torch.nn as nn
import torch.nn.functional as F


class CTJA (nn.Module):
    def __init__(self):
        super(CTJA, self).__init__()
        self.conv_D12 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1))
        self.conv_D13 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2))
        self.conv_D14 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 4, 4), dilation=(1, 4, 4))
        self.bn_D11 = nn.BatchNorm3d(3)
        self.conv_D1D = nn.Conv3d(3, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, groups=3)
        self.bn_D12 = nn.BatchNorm3d(3)
        self.conv_D1P = nn.Conv3d(3, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.bn_D13 = nn.BatchNorm3d(1)

    def forward(self, x):
        T_C = x.size(1) * x.size(2)
        W_H = x.size(3) * x.size(4)
        x_0 = x.view(1, x.size(1), x.size(2), 1, W_H)
        x_0 = x_0.view(1, T_C, 1, 1, W_H)
        x_mean = torch.mean(x, 3, True)
        x_mean = torch.mean(x_mean, 4, True)
        x_mean = x_mean.permute(0, 4, 3, 1, 2)
        x_2 = self.conv_D12(x_mean)
        x_3 = self.conv_D13(x_mean)
        x_4 = self.conv_D14(x_mean)
        x_s = torch.stack([x_2, x_3, x_4], 0)
        x_s = x_s.squeeze(2).permute(1, 0, 2, 3, 4)
        x_s = self.bn_D11(x_s)
        x_s = F.elu(x_s)
        x_s = self.conv_D1D(x_s)
        x_s = self.conv_D1P(x_s)
        x_s = self.bn_D13(x_s)
        x_s = F.elu(x_s)
        x_sig = torch.sigmoid(x_s)
        x_sig = x_sig.view(1, 1, 1, T_C, 1)
        x_sig = x_sig.permute(0, 3, 4, 1, 2)
        x_sig = x_sig.repeat(1, 1, 1, 1, W_H)
        x_m = x_sig.mul(x_0)
        x_m = x_m.view(1, x.size(1), x.size(2), 1, W_H)
        x_m = x_m.view(1, x.size(1), x.size(2), x.size(3), x.size(4))
        
        return x_m


class STJA (nn.Module):
    def __init__(self):
        super(STJA, self).__init__()
        self.conv_st11 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn_st11 = nn.BatchNorm3d(1)
        self.conv_st12 = nn.Conv3d(1, 1, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.bn_st12 = nn.BatchNorm3d(1)

    def forward(self, x):
        x_s = torch.mean(x, 1, True)
        x_s_1 = self.conv_st11(x_s)
        x_s_1 = self.bn_st11(x_s_1)
        x_s_1 = F.elu(x_s_1)
        x_t = torch.mean(torch.mean(x_s, 3, True), 4, True)
        x_t_1 = self.conv_st12(x_t)
        x_t_1 = self.bn_st12(x_t_1)
        x_t_1 = F.elu(x_t_1)
        x_s_sig = torch.sigmoid(x_s_1)
        x_t_sig = torch.sigmoid(x_t_1)
        x_t_sig = x_t_sig.repeat(1, 1, 1, x.size(3), x.size(4))
        x_st_sig = x_s_sig * x_t_sig
        x_st_sig = x_st_sig.repeat(1, x.size(1), 1, 1, 1)
        x = x * x_st_sig
        
        return x


class STJA_2D(nn.Module):
    def __init__(self):
        super(STJA_2D, self).__init__()

        self.conv_st11 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn_st11 = nn.BatchNorm2d(1)
        self.conv_st12 = nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn_st12 = nn.BatchNorm2d(1)

    def forward(self, x):
        x_s = torch.mean(x, 1, True)
        x_s_1 = self.conv_st11(x_s)
        x_s_1 = self.bn_st11(x_s_1)
        x_s_1 = F.elu(x_s_1)
        x_t = torch.mean(torch.mean(x_s, 2, True), 3, True)
        x_t_1 = self.conv_st12(x_t)
        x_t_1 = self.bn_st12(x_t_1)
        x_t_1 = F.elu(x_t_1)
        x_s_sig = torch.sigmoid(x_s_1)
        x_t_sig = torch.sigmoid(x_t_1)
        x_t_sig = x_t_sig.repeat(1, 1, x.size(2), x.size(3))
        x_st_sig = x_s_sig * x_t_sig
        x_st_sig = x_st_sig.repeat(1, x.size(1), 1, 1)
        x = x * x_st_sig
        
        return x
