import torch
from torch import nn
import torch.nn.functional as F
from .JointAttention import CTJA, STJA, STJA_2D


class JAMSNet (nn.Module):
    def __init__(self):
        super(JAMSNet, self).__init__()
        adapt_h = 48
        adapt_w = 32
        T = 150
        # ##############  Multi-scale Feature Extraction & Fusion Net  ###############
        # ###   layer0   ###
        self.conv_00 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.bn_00 = nn.BatchNorm2d(32)
        self.adapt_avg_pool_01 = nn.AdaptiveAvgPool2d((adapt_h, adapt_w))
        self.conv_01 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_01 = nn.BatchNorm2d(32)
        self.conv_02 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_02 = nn.BatchNorm2d(32)
        # ####   layer1   ###
        self.conv_10 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.bn_10 = nn.BatchNorm2d(32)
        self.adapt_avg_pool_11 = nn.AdaptiveAvgPool2d((adapt_h, adapt_w))
        self.conv_11 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_11 = nn.BatchNorm2d(32)
        self.conv_12 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_12 = nn.BatchNorm2d(32)
        # ####   layer2   ###
        self.conv_20 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.bn_20 = nn.BatchNorm2d(32)
        self.adapt_avg_pool_21 = nn.AdaptiveAvgPool2d((adapt_h, adapt_w))
        self.conv_21 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_21 = nn.BatchNorm2d(32)
        self.conv_22 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn_22 = nn.BatchNorm2d(32)

        # ##########################   Layer fuse   ##################################
        self.conv = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0)
        self.adapt_avg_pool_L = nn.AdaptiveAvgPool2d((48, 32))

        # ##########################   rPPG Extraction Net   ##############################
        self.conv_1 = nn.Conv3d(32, 64, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=0)
        self.bn_1 = nn.BatchNorm3d(64)
        # first
        self.max_pool_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_2 = nn.BatchNorm3d(64)
        self.conv_3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_3 = nn.BatchNorm3d(64)
        # second
        self.max_pool_2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.conv_4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_4 = nn.BatchNorm3d(64)
        self.conv_5 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_5 = nn.BatchNorm3d(64)
        # third
        self.max_pool_3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv_6 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_6 = nn.BatchNorm3d(64)
        self.conv_7 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_7 = nn.BatchNorm3d(64)
        # fourth
        self.max_pool_4 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_8 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_8 = nn.BatchNorm3d(64)
        self.conv_9 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn_9 = nn.BatchNorm3d(64)
        # finally
        self.gobal_avg_pool3d = nn.AdaptiveAvgPool3d(output_size=(T, 1, 1))
        self.conv_L = nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)

        self.CTJA_1 = CTJA()
        self.CTJA_2 = CTJA()
        self.CTJA_3 = CTJA()
        self.CTJA_4 = CTJA()
        self.STJA_1 = STJA()
        self.STJA_2 = STJA()
        self.STJA_3 = STJA()
        self.STJA_4 = STJA()
        self.STJA_2D1 = STJA_2D()
        self.STJA_2D2 = STJA_2D()
        self.STJA_2D3 = STJA_2D()

    def forward(self, x0, x1, x2):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x0,x1,x2 ---> T C H W
        # ##############  Multi-scale Feature Extraction & Fusion Net  ###############
        # ###   layer0   ###
        x0 = self.conv_00(x0)
        x0 = self.bn_00(x0)
        x0 = F.elu(x0)
        x0 = self.adapt_avg_pool_01(x0)
        x0 = self.conv_01(x0)
        x0 = self.bn_01(x0)
        x0 = F.elu(x0)
        x0 = self.conv_02(x0)
        x0 = self.STJA_2D1(x0)
        x0 = self.bn_02(x0)
        x0 = F.elu(x0)
        # ####   layer1   ###
        x1 = self.conv_10(x1)
        x1 = self.bn_10(x1)
        x1 = F.elu(x1)
        x1 = self.adapt_avg_pool_11(x1)
        x1 = self.conv_11(x1)
        x1 = self.bn_11(x1)
        x1 = F.elu(x1)
        x1 = self.conv_12(x1)
        x1 = self.STJA_2D2(x1)
        x1 = self.bn_12(x1)
        x1 = F.elu(x1)
        # ####   layer2   ###
        x2 = self.conv_20(x2)
        x2 = self.bn_20(x2)
        x2 = F.elu(x2)
        x2 = self.adapt_avg_pool_21(x2)
        x2 = self.conv_21(x2)
        x2 = self.bn_21(x2)
        x2 = F.elu(x2)
        x2 = self.conv_22(x2)
        x2 = self.STJA_2D3(x2)
        x2 = self.bn_22(x2)
        x2 = F.elu(x2)
        
        # ##########################   Layer fuse   ##################################
        datemean_0 = torch.mean(x0)
        datemean_1 = torch.mean(x1)
        datemean_2 = torch.mean(x2)
        L = torch.zeros((1, 3))
        L[:, 0] = datemean_0
        L[:, 1] = datemean_1
        L[:, 2] = datemean_2
        L = L.reshape(1, 1, 3)
        L = L.to(device)
        L_conv = self.conv(L)
        L_conv = L_conv.squeeze(0)
        L_soft = torch.softmax(L_conv, 1)
        Data = x0 * L_soft[:, 0] + x1 * L_soft[:, 1] + x2 * L_soft[:, 2] 
        Data = self.adapt_avg_pool_L(Data)
        
        # ##########################   rPPG Extraction Net   #############################
        x = Data.unsqueeze(0) 
        x = x.permute(0, 2, 1, 3, 4)  
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.elu(x)
        # first 
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.CTJA_1(x)
        x = self.bn_2(x)
        x = F.elu(x)
        x = self.conv_3(x)
        x = self.STJA_1(x)
        x = self.bn_3(x)
        x = F.elu(x)
        # second
        x = self.max_pool_2(x)
        x = self.conv_4(x)
        x = self.CTJA_2(x)
        x = self.bn_4(x)
        x = F.elu(x)
        x = self.conv_5(x)
        x = self.STJA_2(x)
        x = self.bn_5(x)
        x = F.elu(x)
        # third
        x = self.max_pool_3(x)
        x = self.conv_6(x)
        x = self.CTJA_3(x)
        x = self.bn_6(x)
        x = F.elu(x)
        x = self.conv_7(x)
        x = self.STJA_3(x)
        x = self.bn_7(x)
        x = F.elu(x)
        # fourth
        x = self.max_pool_4(x)
        x = self.conv_8(x)
        x = self.CTJA_4(x)
        x = self.bn_8(x)
        x = F.elu(x)
        x = self.conv_9(x)
        x = self.STJA_4(x)
        x = self.bn_9(x)
        x = F.elu(x)
        # finally
        x = self.gobal_avg_pool3d(x)
        x = self.conv_L(x)
        x = x.squeeze(0).squeeze(0).squeeze(1)

        return x

