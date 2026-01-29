import os
os.environ ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from time import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from glob import glob
from Unet_complex import UNet
from Denoising import Denoising
import pywt
import ptwt
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser



parser = ArgumentParser(description='A2MC-MR-Net')
parser.add_argument('--epoch_num', type=int, default=250, help='epoch number of model')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=6, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--model_dir', type=str, default='A2-MC-MR-Net', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='./M4Raw_data_example', help='training data directory')
parser.add_argument('--training', type=bool, default=False, help='training or testing')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {10, 20, 30, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--lesion_name', type=str, default='whole', help='lesion name')
parser.add_argument('--data_name', type=str, default='M4Raw', help='data name')
parser.add_argument('--segnet_name', type=str, default='unet', help='segnet name')
parser.add_argument('--mask_type', type=str, default='random_central', help='sampling matrix type')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

args = parser.parse_args()
lesion_name = args.lesion_name
segnet_name = args.segnet_name
data_name = args.data_name
model_dir = args.model_dir
epoch_num = args.epoch_num
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
mask_type = args.mask_type
data_path = args.data_dir
batch_size = args.batch_size
training = args.training


try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except:
    pass


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mask = torch.zeros(256, 256)
#pre-trained segmentation model path
path = './Seg_model_M4Raw_whole/net_params_100.pkl'   
# Load CS Sampling Matrix: phi
tmp = sio.loadmat('./mask/central_random_1D_256_mask_10.mat')
initx = tmp['kx']
inity = tmp['ky']
for i in initx[0, :]:
    for j in inity[0, :]:
        mask[i, j] = 1
initx = initx.astype(np.float32)/256
inity = inity.astype(np.float32)/256
initx = torch.from_numpy(initx).to(device)
inity = torch.from_numpy(inity).to(device)
initx = initx.permute(1, 0)
inity = inity.permute(1, 0)
Mum = initx.shape[0]
Num = inity.shape[0]
initxo = initx.expand(Mum, 3).clone()
inityo = inity.expand(Num, 3).clone()

mask = torch.tensor(mask)
mask = mask.view(1, 1, 1, *(mask.shape))

def single_to_concact(x):
    out = torch.cat((x[..., 0], x[..., 1]), 1)
    return out

def concact_to_single(x):
    b, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    num = torch.arange(0, c, 2)
    out = [x[..., 0:2]]
    for i in num[1:]:
        out.append(x[..., i:i+2])
    out = torch.stack(out, dim=1)
    return out

def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_real = torch.real(x)
    x_imag = torch.imag(x)
    y_real = torch.real(y)
    y_imag = torch.imag(y)

    re = x_real * y_real - x_imag * y_imag
    im = x_real * y_imag + x_imag * y_real

    return torch.complex(re, im)

def normalize_tenor(img):
    """
    Normalize the image between 0 and 1
    """
    shp = img.shape
    shp1 = torch.tensor(shp)
    if img.dim() >= 3:
        nimg = torch.prod(shp1[0:-2])
    elif img.dim() == 2:
        nimg = 1
    img = torch.reshape(img, (nimg, shp[-2], shp[-1]))
    eps = 1e-15
    img2 = torch.empty_like(img)

    for i in range(nimg):
        mx = img[i].max()
        mn = img[i].min()
        img2[i] = (img[i]-mn)/(mx-mn+eps)
    img2 = torch.reshape(img2, shp)
    return img2


class data_set(Dataset):
    def __init__(self, data_path, training):
        if training:
            self.data_path = glob(os.path.join(data_path, 'multicoil_train/*.mat'))
        else:
            self.data_path = glob(os.path.join(data_path, 'multicoil_test/*.mat'))
    def __getitem__(self, index):
        image_path = self.data_path[index]
        data = sio.loadmat(image_path)
        coil_map = data['coil_map']
        kspace = data['kspace']
        ground_truth = data['ground_truth']
        seg_label = data['seg_label']
        return coil_map, kspace, ground_truth, seg_label
    def __len__(self):
        return len(self.data_path)

class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()
        self.initx = initxo
        self.inity = nn.Parameter(inityo)

    def forward(self, M, N):
        m = torch.arange(M).to(device)
        n = torch.arange(N).to(device)
        JTwoPi = torch.tensor(1j * 2 * np.pi).to(device)
        scalex = torch.tensor(1. / np.sqrt(M)).to(device)
        scaley = torch.tensor(1. / np.sqrt(N)).to(device)

        orgxT1 = self.initx[:, 0].reshape(1, 1, Mum, 1)
        orgyT1 = self.inity[:, 0].reshape(1, 1, Num, 1)
        maskhT1 = torch.exp(-JTwoPi * (orgxT1 - 1 / 2) * (m - M / 2)) * scalex
        maskvT1 = torch.exp(-JTwoPi * (orgyT1 - 1 / 2) * (n - N / 2)) * scaley
        maskhTT1 = torch.conj(maskhT1.transpose(-1, -2)).to(device)
        maskvTT1 = torch.conj(maskvT1.transpose(-1, -2)).to(device)

        orgxT2 = self.initx[:, 1].reshape(1, 1, Mum, 1)
        orgyT2 = self.inity[:, 1].reshape(1, 1, Num, 1)
        maskhT2 = torch.exp(-JTwoPi * (orgxT2 - 1 / 2) * (m - M / 2)) * scalex
        maskvT2 = torch.exp(-JTwoPi * (orgyT2 - 1 / 2) * (n - N / 2)) * scaley
        maskhTT2 = torch.conj(maskhT2.transpose(-1, -2)).to(device)
        maskvTT2 = torch.conj(maskvT2.transpose(-1, -2)).to(device)

        orgxFlair = self.initx[:, 2].reshape(1, 1, Mum, 1)
        orgyFlair= self.inity[:, 2].reshape(1, 1, Num, 1)
        maskhFlair = torch.exp(-JTwoPi * (orgxFlair - 1 / 2) * (m - M / 2)) * scalex
        maskvFlair = torch.exp(-JTwoPi * (orgyFlair - 1 / 2) * (n - N / 2)) * scaley
        maskhTFlair = torch.conj(maskhFlair.transpose(-1, -2)).to(device)
        maskvTFlair = torch.conj(maskvFlair.transpose(-1, -2)).to(device)

        Maskh = torch.cat((maskhT1, maskhT2, maskhFlair), dim=1)
        MaskhT = torch.cat((maskhTT1, maskhTT2, maskhTFlair), dim=1)
        Maskv = torch.cat((maskvT1, maskvT2, maskvFlair), dim=1)
        MaskvT = torch.cat((maskvTT1, maskvTT2, maskvTFlair), dim=1)

        return [Maskh, MaskhT, Maskv, MaskvT]

Seg_net = UNet()

Seg_net = nn.DataParallel(Seg_net)
model_state_dict = torch.load(path)

class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.1]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.02]))
        self.rho = nn.Parameter(torch.Tensor([0.1]))
        self.miu = nn.Parameter(torch.Tensor([0.1]))
        self.gamma = nn.Parameter(torch.Tensor([0.01]))
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 6, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 32, 3, 3)))
        self.conv4_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 64, 3, 3)))
        self.conv3_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 64, 3, 3)))
        self.conv4_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(6, 32, 3, 3)))
        self.seg = Seg_net
        self.denoising = Denoising()

    def forward(self, x, fft_forback, k, c, batch_size):
        M, N = x.shape[-3], x.shape[-2]
        k = torch.view_as_complex(k)
        csm = torch.view_as_complex(c)
        csmT = torch.conj(csm)
        [Maskh, MaskhT, Maskv, MaskvT] = fft_forback(M, N)
        Maskh = torch.unsqueeze(Maskh, 2)
        maskh = Maskh.expand(k.shape[0], k.shape[1], k.shape[2], Maskh.shape[-2], Maskh.shape[-1]).clone()
        MaskhT = torch.unsqueeze(MaskhT, 2)
        maskhT = MaskhT.expand(k.shape[0], k.shape[1], k.shape[2], MaskhT.shape[-2], MaskhT.shape[-1]).clone()
        Maskv = torch.unsqueeze(Maskv, 2)
        maskv = Maskv.expand(k.shape[0], k.shape[1], k.shape[2], Maskv.shape[-2], Maskv.shape[-1]).clone()
        MaskvT = torch.unsqueeze(MaskvT, 2)
        maskvT = MaskvT.expand(k.shape[0], k.shape[1], k.shape[2], MaskvT.shape[-2], MaskvT.shape[-1]).clone()

        aath = maskhT@maskh
        aatv = maskv.transpose(-1, -2)@maskvT.transpose(-1, -2)

        b, c, h, w, _ = x.shape
        x = x.to(torch.float32)
        x_in = single_to_concact(x)

        _, _, _, z = self.denoising(x_in)

        keymapping = {"module.conv1": self.conv1_forward,
                      "module.conv2": self.conv2_forward,
                      "module.conv3": self.conv3_forward,
                      "module.conv4": self.conv4_forward}
        for key in keymapping:
            model_state_dict[key] = keymapping[key]


        self.seg.load_state_dict(model_state_dict)

        with torch.no_grad():
            seg_pred, _ = self.seg(z)
        # print(seg_pred.requires_grad)

        b, c, h, w = z.shape

        z = concact_to_single(z)
        z = torch.view_as_complex(z)

        x = torch.view_as_complex(x)
        k = maskh @ k @ maskv.transpose(-1, -2)
        x = x.to(torch.complex64)

        x = x - self.lambda_step * torch.sum(complex_mul(csmT, (aath @ (complex_mul(csm, torch.unsqueeze(x, 2)))
                @ aatv)), 2) - self.rho * self.lambda_step * (seg_pred + 1) * (x-z)
        x = x + self.lambda_step * torch.sum(complex_mul(csmT, (maskhT @ k @ maskvT.transpose(-1, -2))), 2)

        x = torch.view_as_real(x)
        b, c, h, w, _ = x.shape
        x_input = single_to_concact(x)

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.conv2d(x, self.conv2_forward, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv3_forward, padding=1)
        x_forward = F.conv2d(x, self.conv4_forward, padding=1)
        #### problem when norm == 0
        x_soft_thr = F.relu(torch.norm(x_forward, p=2, dim=-3) - self.soft_thr) / (torch.norm(x_forward, p=2, dim=-3)+(1e-7))
        #### Soft_threshold Function
        x = torch.mul(x_forward, torch.unsqueeze(x_soft_thr, 1).expand(*(x_forward.shape)))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.conv2d(x, self.conv2_backward, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv3_backward, padding=1)
        x_backward = F.conv2d(x, self.conv4_backward, padding=1)

        x_pred = concact_to_single(x_backward)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.conv2d(x, self.conv2_backward, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv3_backward, padding=1)
        x_D_est = F.conv2d(x, self.conv4_backward, padding=1)
        symloss = x_D_est - x_input
        return [x_pred, symloss, seg_pred]

# Define MC_MR_Net
class MC_MR_Net(torch.nn.Module):
    def __init__(self, LayerNo):
        super(MC_MR_Net, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()
        for i in range(LayerNo):
            onelayer.append(BasicBlock())
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x, k, c, batch_size):
        layers_sym = []   # for computing symmetric loss
        seg_pred = []     # for computing segmentation loss
        for i in range(self.LayerNo):
            [x, layer_sym1, seg_pred1] = self.fcs[i](x, self.fft_forback, k, c, batch_size)
            layers_sym.append(layer_sym1)
            seg_pred.append(seg_pred1)
        x_final = x
        return [x_final, layers_sym, seg_pred]

model = MC_MR_Net(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

Test_labels = data_set(data_path, training)
ImgNum = len(data_path)
print_flag = 1   # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())

def normalize(img):
    """
    Normalize the image between 0 and 1
    """
    shp = img.shape
    if np.ndim(img) >= 3:
        nimg = np.prod(shp[0:-2])
    elif np.ndim(img) == 2:
        nimg = 1
    img = np.reshape(img, (nimg, shp[-2], shp[-1]))
    eps = 1e-15
    img2 = np.empty_like(img)
    for i in range(nimg):
        mx = img[i].max()
        mn = img[i].min()
        img2[i] = (img[i]-mn)/(mx-mn+eps)
    img2 = np.reshape(img2, shp)
    return img2

def seg_loss(pred, label, num):
    seg_loss = dice_loss(pred, label, num)

    return seg_loss

def cross_entropyloss(pred, label, num):
    for i in range(num):
        labeli = label[:, i, :, :]
        predi = pred[:, i, :, :]
        weighted = 1.0 - (torch.sum(labeli) / torch.sum(label))
        if i == 0:
            raw_loss = -1.0 * weighted * labeli * torch.log(torch.clamp(predi, 0.005, 1))
        else:
            raw_loss += -1.0 * weighted * labeli * torch.log(torch.clamp(predi, 0.005, 1))
    loss = torch.sum(raw_loss)

    return loss


def ifft2(kspace):
    image = torch.fft.ifftshift(kspace, dim=(-1, -2))
    image = torch.fft.ifft2(image, dim=(-1, -2))
    image = torch.fft.fftshift(image, dim=(-1, -2))

    return image


def fft2(img):
    kspace = torch.fft.ifftshift(img, dim=(-1, -2))
    kspace = torch.fft.fft2(kspace, dim=(-1, -2))
    kspace = torch.fft.fftshift(kspace, dim=(-1, -2))

    return kspace


def dice_loss(pred, label, num):
    dice = 0.0
    for j in range(batch_size):
        for i in range(num):
            inse = torch.sum(pred[j, i, :, :] * label[j, i, :, :])
            l = torch.sum(pred[j, i, :, :])
            r = torch.sum(label[j, i, :, :])
            dice += (2.0 * inse + 1)/(l + r + 1)

    return 1.0 - 1.0 * dice / num / batch_size

model_dir = "./%s/MRI_MC_MR_Net_layer_%d_group_%d_cs_ratio_%d_%s" % (model_dir, layer_num, group_num, cs_ratio, mask_type)
# Load pre-trained model with epoch number
model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch_num)))


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def label_ssim(img1, img2, label):
    nonzero_indices = np.argwhere(label)
    if len(nonzero_indices) > 0:
        min_x, min_y = nonzero_indices.min(axis=0)
        max_x, max_y = nonzero_indices.max(axis=0)
        img1 = img1 * label
        img2 = img2 * label

        img1 = img1[min_x:max_x+11, min_y:max_y + 11]
        img2 = img2[min_x:max_x+11, min_y:max_y + 11]
        return ssim(img1, img2, data_range=255, win_size=3)
    else:
        return ssim(img1, img2, data_range=255, win_size=3)


def label_psnr(img1, img2, label):
    img1.astype(np.float32)
    img2.astype(np.float32)
    num = np.sum(label)
    label = (label == 1)
    img1 = img1[label]
    img2 = img2[label]
    mse = np.sum((img1 - img2) ** 2)/(num+1e-6)
    if mse == 0:
        return 100
    PIXEL_img2 = max(img2)
    PIXEL_MAX = PIXEL_img2
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


result_dir = os.path.join(args.result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

PSNR_All = np.zeros((ImgNum, 3), dtype=np.float32)
SSIM_All = np.zeros((ImgNum, 3), dtype=np.float32)

PSNR_de_All = np.zeros((ImgNum, 3), dtype=np.float32)
SSIM_de_All = np.zeros((ImgNum, 3), dtype=np.float32)

PSNR_bc_All = np.zeros((ImgNum, 3), dtype=np.float32)
SSIM_bc_All = np.zeros((ImgNum, 3), dtype=np.float32)

Init_PSNR_All = np.zeros((ImgNum, 3), dtype=np.float32)
Init_SSIM_All = np.zeros((ImgNum, 3), dtype=np.float32)

Init_de_PSNR_All = np.zeros((ImgNum, 3), dtype=np.float32)
Init_de_SSIM_All = np.zeros((ImgNum, 3), dtype=np.float32)

Init_bc_PSNR_All = np.zeros((ImgNum, 3), dtype=np.float32)
Init_bc_SSIM_All = np.zeros((ImgNum, 3), dtype=np.float32)

O_PSNR_All = np.zeros((ImgNum, 3), dtype=np.float32)
O_SSIM_All = np.zeros((ImgNum, 3), dtype=np.float32)


print('\n')
print("MRI CS Reconstruction Start")


def normalize(img):
    """
    Normalize the image between 0 and 1
    """
    shp = img.shape
    if np.ndim(img) >= 3:
        nimg = np.prod(shp[0:-2])
    elif np.ndim(img) == 2:
        nimg = 1
    img = np.reshape(img, (nimg, shp[-2], shp[-1]))
    eps = 1e-15
    img2 = np.empty_like(img)
    for i in range(nimg):
        mx = img[i].max()
        mn = img[i].min()
        img2[i] = (img[i]-mn)/(mx-mn+eps)
    img2 = np.reshape(img2, shp)
    return img2

def normalize_tenor(img):
    """
    Normalize the image between 0 and 1
    """
    shp = img.shape
    shp1 = torch.tensor(shp)
    if img.dim() >= 3:
        nimg = torch.prod(shp1[0:-2])
    elif img.dim() == 2:
        nimg = 1
    img = torch.reshape(img, (nimg, shp[-2], shp[-1]))
    eps = 1e-15
    img2 = torch.empty_like(img)

    for i in range(nimg):
        mx = img[i].max()
        mn = img[i].min()
        img2[i] = (img[i]-mn)/(mx-mn+eps)
    img2 = torch.reshape(img2, shp)
    return img2

def complex_normalize(img):
    real = torch.real(img)
    real = normalize_tenor(real)
    imag = torch.imag(img)
    imag = normalize_tenor(imag)
    normalize_img = torch.complex(real, imag)

    return normalize_img


with torch.no_grad():
    for img_no in tqdm(range(ImgNum)):
        coil_map, kspace, ground_truth, seg_label = Test_labels[img_no]
        seg_label = seg_label[0, 0, :, :]
        # ## whole
        # seg_label_cordinate = np.isin(seg_label, np.arange(1, 117))
        # seg_label = seg_label_cordinate.astype(np.int32)

        ## subcortical
        seg_label = np.where(
            (seg_label == 71) | (seg_label == 72) | (seg_label == 73) | (seg_label == 74) | (seg_label == 75) |
            (seg_label == 76) | (seg_label == 77) | (seg_label == 78) | (seg_label == 37) | (seg_label == 38) |
            (seg_label == 39) | (seg_label == 40) | (seg_label == 41) | (seg_label == 42) | (seg_label == 91) |
            (seg_label == 92) | (seg_label == 93) | (seg_label == 94) | (seg_label == 95) | (seg_label == 96) |
            (seg_label == 97) | (seg_label == 98) | (seg_label == 99) | (seg_label == 100) | (seg_label == 101) |
            (seg_label == 102) | (seg_label == 103) | (seg_label == 104) | (seg_label == 105) | (seg_label == 106) |
            (seg_label == 107) | (seg_label == 108) | (seg_label == 109) | (seg_label == 110) | (seg_label == 111) |
            (seg_label == 112) | (seg_label == 113) | (seg_label == 114) | (seg_label == 115) | (seg_label == 116)
            , 1, 0)
        bc_label = (-1) * (seg_label - 1)

        start = time()
        coil_map = torch.from_numpy(coil_map)
        coil_map = coil_map.unsqueeze(0)
        kspace = torch.from_numpy(kspace)
        kspace = kspace.unsqueeze(0)
        coil_map = coil_map.to(device)
        kspace = kspace.to(device)
        mask = mask.to(device)
        kspace_img = ifft2(kspace)
        masked_x_in_k_space = kspace * mask.expand(*(kspace.shape))
        img = ifft2(masked_x_in_k_space)
        img = img.sum(2)

        initial_image = torch.view_as_real(img)
        initial_image = initial_image.to(device)
        [x_output, loss_layers_sym, seg_pred] = model(initial_image, kspace_img, coil_map, batch_size)
        x_output = torch.view_as_complex(x_output.contiguous())

        end = time()

        X_init = img.cpu().data.numpy().reshape(3, img.shape[-2], img.shape[-1])

        Prediction_value = x_output.cpu().data.numpy().reshape(3, x_output.shape[-2], x_output.shape[-1])

        X_init = np.abs(X_init)
        X_init = normalize(X_init)
        X_rec = np.abs(Prediction_value)
        X_rec = normalize(X_rec)
        ground_truth = np.abs(ground_truth)
        ground_truth = normalize(ground_truth)

        for i in range(3):
            init_PSNR = psnr(X_init[i] * 255, 255 * ground_truth[i])
            init_SSIM = ssim(X_init[i] * 255, 255 * ground_truth[i], data_range=255)

            init_dePSNR = label_psnr(X_init[i] * 255, 255 * ground_truth[i], seg_label)
            init_deSSIM = label_ssim(X_init[i] * 255, 255 * ground_truth[i], seg_label)

            init_bcPSNR = label_psnr(X_init[i] * 255, 255 * ground_truth[i], bc_label)
            init_bcSSIM = label_ssim(X_init[i] * 255, 255 * ground_truth[i], bc_label)

            rec_PSNR = psnr(X_rec[i] * 255, 255 * ground_truth[i])
            rec_SSIM = ssim(X_rec[i] * 255, 255 * ground_truth[i], data_range=255)

            rec_de_PSNR = label_psnr(X_rec[i] * 255, 255 * ground_truth[i], seg_label)
            rec_de_SSIM = label_ssim(X_rec[i] * 255, 255 * ground_truth[i], seg_label)

            rec_bc_PSNR = label_psnr(X_rec[i] * 255, 255 * ground_truth[i], bc_label)
            rec_bc_SSIM = label_ssim(X_rec[i] * 255, 255 * ground_truth[i], bc_label)

            PSNR_All[img_no, i] = rec_PSNR
            SSIM_All[img_no, i] = rec_SSIM

            PSNR_de_All[img_no, i] = rec_de_PSNR
            SSIM_de_All[img_no, i] = rec_de_SSIM

            PSNR_bc_All[img_no, i] = rec_bc_PSNR
            SSIM_bc_All[img_no, i] = rec_bc_SSIM

            Init_PSNR_All[img_no, i] = init_PSNR
            Init_SSIM_All[img_no, i] = init_SSIM

            Init_de_PSNR_All[img_no, i] = init_dePSNR
            Init_de_SSIM_All[img_no, i] = init_deSSIM

            Init_bc_PSNR_All[img_no, i] = init_bcPSNR
            Init_bc_SSIM_All[img_no, i] = init_bcSSIM

            PSNR1 = psnr(ground_truth[i] * 255, 255 * ground_truth[i])
            SSIM1 = ssim(ground_truth[i] * 255, 255 * ground_truth[i], data_range=255)
            O_PSNR_All[img_no, i] = PSNR1
            O_SSIM_All[img_no, i] = SSIM1


        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        # 第一列：X_init
        for i, ax in enumerate(axes[:, 0]):
            ax.imshow(X_init[i], cmap='gray')
            ax.text(0.5, -0.04, f'PSNR/TOI-PSNR: {Init_PSNR_All[img_no, i]:.2f}/{Init_de_PSNR_All[img_no, i]:.2f}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
        # 第二列：X_rec
        for i, ax in enumerate(axes[:, 1]):
            ax.imshow(X_rec[i], cmap='gray')
            ax.text(0.5, -0.04, f'PSNR/TOI-PSNR: {PSNR_All[img_no, i]:.2f}/{PSNR_de_All[img_no, i]:.2f}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
        # 第三列：seg_label
        for i, ax in enumerate(axes[:, 2]):
            ax.imshow(ground_truth[i], cmap='gray')
            ax.text(0.5, -0.04, f'Ground_truth',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
        # 第四列：seg_label
        axes[0, 3].imshow(seg_label, cmap='gray')
        axes[0, 3].text(0.5, -0.04, f'img_no: {img_no}',
                        ha='center', va='center', transform=axes[0, 3].transAxes, fontsize=10)
        axes[0, 3].axis('off')
        # 隐藏多余的子图
        for ax in axes[1:, 3]:
            ax.axis('off')
        # 调整布局
        plt.tight_layout(pad=0.1)
        plt.show()

        sio.savemat('./%s/%s_%s_%s_%s_%s_%d_%d.mat' % (model_dir, lesion_name, segnet_name, data_name, args.model_dir, cs_ratio, layer_num, img_no),
            {'init': X_init, 'rec': X_rec, 'ground_truth': ground_truth, 'seg_label':seg_label, 'PSNR': PSNR_All[img_no], 'SSIM':SSIM_All, 'PSNR_de':PSNR_de_All})

        del x_output

    print('\n')
    Re_PSNR = np.mean(PSNR_All, axis=0)
    Re_SSIM = np.mean(SSIM_All, axis=0)
    Re_de_PSNR = np.mean(PSNR_de_All, axis=0)
    Re_de_SSIM = np.mean(SSIM_de_All, axis=0)
    Re_bg_PSNR = np.mean(PSNR_bc_All, axis=0)
    Re_bg_SSIM = np.mean(SSIM_bc_All, axis=0)
    
    print("init PSNR", np.mean(Init_PSNR_All, axis=0), "init SSIM", np.mean(Init_SSIM_All, axis=0))
    print("init disease PSNR", np.mean(Init_de_PSNR_All, axis=0), "init SSIM", np.mean(Init_de_SSIM_All, axis=0))
    print("init background PSNR", np.mean(Init_bc_PSNR_All, axis=0), "init SSIM", np.mean(Init_bc_SSIM_All, axis=0))
    print("Reconstruct PSNR", np.mean(PSNR_All, axis=0), "Reconstruct SSIM", np.mean(SSIM_All, axis=0))
    print("Reconstruct disease PSNR", np.mean(PSNR_de_All, axis=0), "Reconstruct SSIM", np.mean(SSIM_de_All, axis=0))
    print("Reconstruct background PSNR", np.mean(PSNR_bc_All, axis=0), "Reconstruct SSIM", np.mean(SSIM_bc_All, axis=0))
    print("T1", Re_PSNR[0], Re_de_PSNR[0], Re_bg_PSNR[0], Re_SSIM[0], Re_de_SSIM[0], Re_bg_SSIM[0])
    print("T2", Re_PSNR[1], Re_de_PSNR[1], Re_bg_PSNR[1], Re_SSIM[1], Re_de_SSIM[1], Re_bg_SSIM[1])
    print("Flair", Re_PSNR[2], Re_de_PSNR[2], Re_bg_PSNR[2], Re_SSIM[2], Re_de_SSIM[2], Re_bg_SSIM[2])

    data_to_save = {
        'T1': [Re_PSNR[0], Re_de_PSNR[0], Re_bg_PSNR[0], Re_SSIM[0], Re_de_SSIM[0], Re_bg_SSIM[0]],
        'T2': [Re_PSNR[1], Re_de_PSNR[1], Re_bg_PSNR[1], Re_SSIM[1], Re_de_SSIM[1], Re_bg_SSIM[1]],
        'Flair': [Re_PSNR[2], Re_de_PSNR[2], Re_bg_PSNR[2], Re_SSIM[2], Re_de_SSIM[2], Re_bg_SSIM[2]]
    }

    path = '%s/%s_%s_%s_%s_%s_%d.txt' % (model_dir, lesion_name, segnet_name, data_name, args.model_dir, cs_ratio, layer_num)
    with open(path, 'w') as file:
        for key, value in data_to_save.items():
            file.write(f'{key}: {value}\n')

    print("A2MC-MRI Reconstruction End")