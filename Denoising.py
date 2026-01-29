import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Denoising(torch.nn.Module):
    def __init__(self):
        super(Denoising, self).__init__()
        self.conv1_encoder = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 6, 3, 3)))
        self.conv2_encoder = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_encoder = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 32, 3, 3)))
        self.conv4_encoder = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv5_encoder = nn.Parameter(init.xavier_normal_(torch.Tensor(128, 64, 3, 3)))
        self.conv6_encoder = nn.Parameter(init.xavier_normal_(torch.Tensor(256, 128, 3, 3)))
        self.conv7_encoder = nn.Parameter(init.xavier_normal_(torch.Tensor(128, 256, 3, 3)))

        self.conv1_Decoder = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 128, 3, 3)))
        self.conv2_Decoder = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 128, 3, 3)))
        self.conv3_Decoder = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv4_Decoder = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 64, 3, 3)))
        self.conv5_Decoder = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 64, 3, 3)))
        self.conv6_Decoder = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv7_Decoder = nn.Parameter(init.xavier_normal_(torch.Tensor(6, 32, 3, 3)))

    def forward(self, x):
        e1_c1 = F.conv2d(x, self.conv1_encoder, padding=1)
        e1_c1 = nn.InstanceNorm2d(e1_c1.shape[1], eps=1e-6)(e1_c1)
        e1_c1 = F.relu(e1_c1)

        e1_c2 = F.conv2d(e1_c1, self.conv2_encoder, padding=1)
        e1_c2 = nn.InstanceNorm2d(e1_c2.shape[1], eps=1e-6)(e1_c2)
        e1_c2 = F.relu(e1_c2)

        e2_c1 = F.avg_pool2d(e1_c2, kernel_size=2)
        e2_c1 = F.conv2d(e2_c1, self.conv3_encoder, padding=1)
        e2_c1 = nn.InstanceNorm2d(e2_c1.shape[1], eps=1e-6)(e2_c1)
        e2_c1 = F.relu(e2_c1)

        e2_c2 = F.conv2d(e2_c1, self.conv4_encoder, padding=1)
        e2_c2 = nn.InstanceNorm2d(e2_c2.shape[1], eps=1e-6)(e2_c2)
        e2_c2 = F.relu(e2_c2)

        e3_c1 = F.avg_pool2d(e2_c2, kernel_size=2)
        e3_c1 = F.conv2d(e3_c1, self.conv5_encoder, padding=1)
        e3_c1 = nn.InstanceNorm2d(e3_c1.shape[1], eps=1e-6)(e3_c1)
        e3_c1 = F.relu(e3_c1)

        e3_c2 = F.conv2d(e3_c1, self.conv6_encoder, padding=1)
        e3_c2 = nn.InstanceNorm2d(e3_c1.shape[1], eps=1e-6)(e3_c2)
        e3_c2 = F.relu(e3_c2)

        e3_c2 = F.conv2d(e3_c2, self.conv7_encoder, padding=1)
        e3_c2 = nn.InstanceNorm2d(e3_c2.shape[1], eps=1e-6)(e3_c2)
        e3_c2 = F.relu(e3_c2)

        d2 = F.interpolate(e3_c2, scale_factor=2, mode='nearest')
        d2 = F.conv2d(d2, self.conv1_Decoder, padding=1)

        d2_c1 = torch.cat((d2, e2_c2), 1)
        d2_c1 = F.conv2d(d2_c1, self.conv2_Decoder, padding=1)
        d2_c1 = nn.InstanceNorm2d(d2_c1.shape[1], eps=1e-6)(d2_c1)
        d2_c1 = F.relu(d2_c1)

        d2_c2 = F.conv2d(d2_c1, self.conv3_Decoder, padding=1)
        d2_c2 = nn.InstanceNorm2d(d2_c2.shape[1], eps=1e-6)(d2_c2)
        d2_c2 = F.relu(d2_c2)

        d1 = F.interpolate(d2_c2, scale_factor=2, mode='nearest')
        d1 = F.conv2d(d1, self.conv4_Decoder, padding=1)

        d1_c1 = torch.cat((d1, e1_c2), 1)
        d1_c1 = F.conv2d(d1_c1, self.conv5_Decoder, padding=1)
        d1_c1 = nn.InstanceNorm2d(d1_c1.shape[1], eps=1e-6)(d1_c1)
        d1_c1 = F.relu(d1_c1)

        d1_c2 = F.conv2d(d1_c1, self.conv6_Decoder, padding=1)
        d1_c2 = nn.InstanceNorm2d(d1_c2.shape[1], eps=1e-6)(d1_c2)
        d1_c2 = F.relu(d1_c2)

        d1_c3 = F.conv2d(d1_c2, self.conv7_Decoder, padding=1)

        out = d1_c3 + x

        return e1_c2, e2_c2, e3_c2, out
