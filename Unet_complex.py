import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Conv(torch.nn.Module):
    def __init__(self, int_channel, out_channel):
        super(Conv, self).__init__()
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(out_channel, int_channel, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(out_channel, out_channel, 3, 3)))
        self.batchnorm1 = nn.BatchNorm2d(out_channel)
        self.batchnorm2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = F.conv2d(x, self.conv1, padding=1)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = F.conv2d(x, self.conv2, padding=1)
        x = self.batchnorm2(x)
        x = F.relu(x)
        return x


class DownSampling(torch.nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(x, kernel_size=2)
        return x

class UpSampling(torch.nn.Module):
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.conv_UpSampling = nn.Parameter(init.xavier_normal_(torch.Tensor(C//2, C, 3, 3)))

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = F.conv2d(up, self.conv_UpSampling, padding=1)

        return torch.cat((x, r), 1)


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 6, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 32, 3, 3)))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.C1 = Conv(64, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.conv_seg = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 64, 1, 1)))

    def forward(self, x):
        x = F.conv2d(x, self.conv1, padding=1)
        x = F.conv2d(x, self.conv2, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv3, padding=1)
        x = F.conv2d(x, self.conv4, padding=1)
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        seg_logit = F.conv2d(O4, self.conv_seg)

        seg_pred = F.softmax(seg_logit, 1)
        seg_pred = torch.unsqueeze(seg_pred[:, 1, :, :], 1)
        seg_logit = torch.unsqueeze(seg_logit[:, 1, :, :], 1)
        return seg_pred, seg_logit