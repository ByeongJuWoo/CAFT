import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
from ..train_util import *
import math

# define the function includes in import *
__all__ = [
    'SpixelNet1l','SpixelNet1l_bn','SFCN', 'SFCN_small'
]


class SpixelNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(SpixelNet,self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9
        self.patch_size = 16

        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.pred_mask3 = predict_mask(128, self.assign_ch)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.pred_mask1 = predict_mask(32, self.assign_ch)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32 , 16)
        self.pred_mask0 = predict_mask(16,self.assign_ch)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x)) #5*5
        out2 = self.conv1b(self.conv1a(out1)) #11*11
        out3 = self.conv2b(self.conv2a(out2)) #23*23
        out4 = self.conv3b(self.conv3a(out3)) #47*47
        out5 = self.conv4b(self.conv4a(out4)) #95*95

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)
        prob0 = self.softmax(mask0)

        return prob0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
    
class SpixelNetCustom(nn.Module):
    expansion = 1

    def __init__(self, n_spix=196, batchNorm=True):
        super(SpixelNetCustom, self).__init__()

        OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
        OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        self.n_spix=n_spix
        self.batchNorm = batchNorm
        self.assign_ch = 9
        self.patch_size = 16

        self.prev_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)  
        self.prev_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.new_mean = torch.tensor([0.411, 0.432, 0.45]).view(1, 3, 1, 1)

        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.pred_mask3 = predict_mask(128, self.assign_ch)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.pred_mask1 = predict_mask(32, self.assign_ch)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32 , 16)
        self.pred_mask0 = predict_mask(16,self.assign_ch)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x, return_slic_loss=False):
        x = self.re_normalize(x)
        out1 = self.conv0b(self.conv0a(x)) #5*5
        out2 = self.conv1b(self.conv1a(out1)) #11*11
        out3 = self.conv2b(self.conv2a(out2)) #23*23
        out4 = self.conv3b(self.conv3a(out3)) #47*47
        out5 = self.conv4b(self.conv4a(out4)) #95*95

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)
        prob0 = self.softmax(mask0)

        if return_slic_loss:
            loss_col, loss_pos = self.loss(x, prob0)
            return prob0, loss_col, loss_pos

        return prob0
    
    def loss(self, x, y):
        m = 0.03 #0.003 # positional weight
        sp_w = 0.1

        target_feat = self.build_target_feat(x)
        pooled_feat = poolfeat(target_feat, y, self.patch_size, self.patch_size)
        recon_feat = upfeat(pooled_feat, y, self.patch_size, self.patch_size)
        recon_col_feat, target_col_feat = recon_feat[:, :-2, :, :], target_feat[:, :-2, :, :]
        recon_pos_feat, target_pos_feat = recon_feat[:, -2:, :, :], target_feat[:, -2:, :, :]
        
        # positional loss for sfcn
        criterion_mse = torch.nn.MSELoss()
        loss_pos = criterion_mse(recon_pos_feat, target_pos_feat) * (m / self.patch_size)

        # color loss for sfcn
        loss_col = criterion_mse(recon_col_feat, target_col_feat) 
        loss_pos, loss_col = sp_w * loss_pos, sp_w * loss_col
        
        return loss_col, loss_pos

    def build_xy_feat(self, b, h, w, device):
        y_coords = torch.arange(h, dtype=torch.float32, device=device).view(1, h, 1).expand(b, -1, w)
        x_coords = torch.arange(w, dtype=torch.float32, device=device).view(1, 1, w).expand(b, h, -1)
        xy_coord = torch.stack((x_coords, y_coords), dim=1)
        return xy_coord
    
    def rgb2Lab_torch(self, img_in, 
                      mean_values = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
                      std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)):
        # input img should be [0,1] float b*3*h*w
        img= (img_in.clone() * std.to(img_in.device) + mean_values.to(img_in.device)).clamp(0,1)
        mask = img > 0.04045
        img[mask] = torch.pow((img[mask] + 0.055) / 1.055, 2.4)
        img[~mask] /= 12.92
        xyz_from_rgb = torch.tensor([[0.412453, 0.357580, 0.180423],
                                [0.212671, 0.715160, 0.072169],
                                [0.019334, 0.119193, 0.950227]]).to(img_in.device)
        rgb = img.permute(0,2,3,1)
        xyz_img = torch.matmul(rgb, xyz_from_rgb.transpose_(0,1))
        xyz_ref_white = torch.tensor([0.95047, 1., 1.08883]).to(img_in.device)

        # scale by CIE XYZ tristimulus values of the reference white point
        lab = xyz_img / xyz_ref_white

        # Nonlinear distortion and linear transformation
        mask = lab > 0.008856
        lab[mask] = torch.pow(lab[mask], 1. / 3.)
        lab[~mask] = 7.787 * lab[~mask] + 16. / 116.
        x, y, z = lab[..., 0:1], lab[..., 1:2], lab[..., 2:3]

        # Vector scaling
        L = (116. * y) - 16.
        a = 500.0 * (x - y)
        b = 200.0 * (y - z)
        return torch.cat([L, a, b], dim=-1).permute(0,3,1,2)
    
    def seg_gt_to_one_hot(self, seg_gt, num_classes=19, ignore_index=255):
        # valid_mask = (seg_gt != ignore_index).unsqueeze(1) # (B, 1, H, W)

        # Convert ignore_index to 0 """temporarily""" to avoid one_hot() error
        seg_gt_clamped = seg_gt.clone()
        seg_gt_clamped[seg_gt == ignore_index] = 0

        one_hot = F.one_hot(seg_gt_clamped, num_classes=num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        return one_hot
    
    def build_target_feat(self, im):
        with torch.no_grad():
            b, _, h, w = im.shape
            xy_feat = self.build_xy_feat(b, h, w, im.device)
            color_feat = self.rgb2Lab_torch(im).type(torch.float)
            target_feat = torch.cat([color_feat, xy_feat], dim=1)
        return target_feat

    
    def get_hard_label(self, x, n_spix = None):
        B, _, H, W = x.shape
        n_spix = n_spix or self.n_spix
        sw = int(math.sqrt(n_spix * W / H)) 
        sh = int(math.sqrt(n_spix * H / W)) 
        h, w = H//sh, W//sw 
        H_, W_  = int(np.ceil(H/h)*h), int(np.ceil(W/w)*w)
        x = F.interpolate(x, size=(H_, W_), mode='bicubic', align_corners=False)
        output = self(x)
        return self._get_hard_label(output, H_, W_)
    
    def _get_hard_label(self, output, H_, W_, n_spix = None):
        n_spix = n_spix or self.n_spix
        sw = int(math.sqrt(n_spix * W_ / H_)) 
        sh = int(math.sqrt(n_spix * H_ / W_)) 
        h, w = H_//sh, W_//sw 
        spix_values = np.int32(np.arange(0, sw * sh).reshape((sh, sw)))
        spix_idx_tensor_ = shift9pos(spix_values)
        spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, h, axis=1), w, axis=2)
        spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).to(output.device)
        curr_spixl_map = update_spixl_map(spixeIds, output)
        ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_,W_), mode='nearest').type(torch.int).squeeze(1)
        ori_sz_spixel_map = ori_sz_spixel_map.to(dtype=torch.int64)
        torch.set_printoptions(threshold=float('inf'))
        if ori_sz_spixel_map.max() > 2303:
            print(f"sh, sw, h, w: ",sh, sw, h, w )
            # print(spixeIds.shape) # [1, 9, 768, 768]
            # print(output.shape) # [B, 9, H_, W_]
            # print(ori_sz_spixel_map.shape) # [B, H_, W_]
            print(ori_sz_spixel_map.min())
            print(ori_sz_spixel_map.max())
            # print(output[0][:, H_-1, :])
            print(torch.nonzero(ori_sz_spixel_map[0] > 2303, as_tuple=False))
            # print(ori_sz_spixel_map[0][H_-1, :])
            print("--------------------")

        return ori_sz_spixel_map
    
    def re_normalize(self, x):
        # we will not use this. just following imagenet mean and std
        x = x * self.prev_std.to(x.device) + self.prev_mean.to(x.device)  
        x = x - self.new_mean.to(x.device)  
        return x

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
    
class SpixelNetCustomSmall(nn.Module):
    expansion = 1

    def __init__(self, n_spix=196, batchNorm=False):
        super(SpixelNetCustomSmall, self).__init__()

        self.n_spix=n_spix
        self.batchNorm = False
        self.assign_ch = 9

        self.prev_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  
        self.prev_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.new_mean = torch.tensor([0.411, 0.432, 0.45]).view(1, 3, 1, 1)

        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.conv4a = conv(self.batchNorm, 128, 128, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.deconv3 = deconv(128, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        # self.pred_mask3 = predict_mask(128, self.assign_ch)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        # self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        # self.pred_mask1 = predict_mask(32, self.assign_ch)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32 , 16)
        self.pred_mask0 = predict_mask(16,self.assign_ch)
        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x)) #5*5
        out2 = self.conv1b(self.conv1a(out1)) #11*11
        out3 = self.conv2b(self.conv2a(out2)) #23*23
        out4 = self.conv3b(self.conv3a(out3)) #47*47
        out5 = self.conv4b(self.conv4a(out4)) #95*95

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)  # torch.Size([4, 256, 40, 56])
        out_conv3_1 = self.conv3_1(concat3)  # torch.Size([4, 128, 40, 56])

        out_deconv2 = self.deconv2(out_conv3_1)  # 64 80
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)
        prob0 = self.softmax(mask0)

        return prob0
    
    def get_hard_label(self, x):
        B, _, H, W = x.shape
        sw = int(math.sqrt(self.n_spix * W / H)) 
        sh = int(math.sqrt(self.n_spix * H / W)) 
        h, w = H//sh, W//sw 
        H_, W_  = int(np.ceil(H/h)*h), int(np.ceil(W/w)*w)
        x = F.interpolate(x, size=(H_, W_), mode='bicubic', align_corners=False)
        output = self(x)
        return self._get_hard_label(output, H_, W_)
    
    def _get_hard_label(self, output, H_, W_):
        sw = int(math.sqrt(self.n_spix * W_ / H_)) 
        sh = int(math.sqrt(self.n_spix * H_ / W_)) 
        h, w = H_//sh, W_//sw 
        spix_values = np.int32(np.arange(0, sw * sh).reshape((sh, sw)))
        spix_idx_tensor_ = shift9pos(spix_values)
        spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, h, axis=1), w, axis=2)
        spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).to(output.device)
        curr_spixl_map = update_spixl_map(spixeIds, output)
        ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_,W_), mode='nearest').type(torch.int).squeeze(1)
        ori_sz_spixel_map = ori_sz_spixel_map.to(dtype=torch.int64)

        return ori_sz_spixel_map
    
    def re_normalize(self, x):
        # we will not use this. just following imagenet mean and std
        x = x * self.prev_std.to(x.device) + self.prev_mean.to(x.device)  
        x = x - self.new_mean.to(x.device)  
        return x

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def SpixelNet1l( data=None):
    # Model without  batch normalization
    model = SpixelNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def SpixelNet1l_bn(data=None):
    # model with batch normalization
    model = SpixelNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

def SFCN(n_spix=196, data=None):
    # model with batch normalization
    model = SpixelNetCustom(n_spix=n_spix, batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

def SFCN_small(n_spix=196, data=None):
    # model with batch normalization
    model = SpixelNetCustomSmall(n_spix=n_spix, batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
#
