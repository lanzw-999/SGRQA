import torchvision
import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import numpy as np
import math
from instance_whitening import cross_whitening_loss, InstanceWhitening

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=1, channels=None):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        b0 = self.relu(self.bn1(self.conv1(x)))
        b = self.maxpool(b0)
        b1 = self.layer1(b)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )



class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )



class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, res], dim=1)
        x = self.post_conv(x)
        return x




class E_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.1):
        super(E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=ksize,
                                groups=hidden_features)
        self.conv2 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                                groups=hidden_features)
        self.conv3 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=1,
                                groups=hidden_features)


        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x3 = self.conv3(x)

        x = self.fc2(x1 + x2 + x3)
        x = self.drop(self.act(x))

        return x



class Self_Attention(nn.Module):
    """ Self attention Layer"""

    def __init__(self,in_dim):
        super(Self_Attention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=3, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.dleta = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.mlp = E_FFN(in_features=in_dim, hidden_features=in_dim // 4, out_features=in_dim // 4, act_layer=nn.ReLU6, drop=0)


    def forward(self, x ,cancha):
        """
        inputs :
            x : input feature maps (B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(cancha).view(m_batchsize, -1, width*height).permute(0,2,1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        out = self.mlp(out)

        return out

class cAttention(nn.Module):  # 通道注意力
    def __init__(self, in_dim, out_dim, fc_ratio):
        super(cAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // fc_ratio, 1, 1),
            nn.ReLU6()
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_dim // fc_ratio, out_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        max_out = self.fc1(self.max_pool(x))
        out = avg_out + max_out
        return self.fc2(out)


class MaxMeanPool(nn.Module):
    def __init__(self):
        super(MaxMeanPool, self).__init__()

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)

        return s_attn



class MAF(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, in_dim, out_dim, dilation=2):
        super(MAF, self).__init__()

        self.ac = cAttention(in_dim, out_dim, 4)
        self.mm = MaxMeanPool()

        self.dilated_conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=dilation, dilation=dilation)

        self.sigma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        c_c = self.ac(x)
        mm_c = self.mm(x)
        map = c_c * mm_c

        attn1 = self.dilated_conv(x)

        #attn2 = self.dilated_conv2(x)

        attn = attn1 + self.sigma * map

        return attn


class Net(nn.Module):
    def __init__(self,backbone=ResNet50):
        super().__init__()

        self.backbone = backbone()

        self.res_backbone = backbone()

        self.conv3 = MAF(1024, 512)
        self.conv2 = MAF(512, 256)
        self.conv1 = MAF(256, 128)

        self.self_attention1 = Self_Attention(256)
        self.self_attention2 = Self_Attention(512)
        self.self_attention3 = Self_Attention(1024)
        self.self_attention4 = Self_Attention(2048)


        self.L2pooling_l1 = L2pooling(channels=256)
        self.L2pooling_l2 = L2pooling(channels=512)
        self.L2pooling_l3 = L2pooling(channels=1024)
        self.L2pooling_l4 = L2pooling(channels=2048)

        self.avg7 = nn.AvgPool2d((7, 7))
        self.avg8 = nn.AvgPool2d((8, 8))
        self.avg4 = nn.AvgPool2d((4, 4))
        self.avg2 = nn.AvgPool2d((2, 2))

        self.fc2 = nn.Linear(3840, 1024)
        self.fc1 = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, 1)

        self.drop2d = nn.Dropout(p=0.1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Fusion3 = Fusion(1024)
        self.Fusion2 = Fusion(512)
        self.Fusion1 = Fusion(256)

    def forward(self, x, residual, reference):
        res1, res2, res3, res4 = self.backbone(x)
        res11, res22, res33, res44 = self.res_backbone(reference)

        k_arr = [res1, res2, res3, res4]
        q_arr = [res11, res22, res33, res44]

        CCL = torch.FloatTensor([0]).cuda()

        for N, f_maps in enumerate(zip(k_arr, q_arr)):
            k_maps, q_maps = f_maps
            # detach original images
            q_maps = q_maps.detach()

            crosscov_loss = cross_whitening_loss(k_maps, q_maps)
            CCL = CCL + crosscov_loss
        CCL = CCL / len(k_arr)

        residual = self.maxpool(residual)
        residual1 = self.maxpool(residual)
        residual2 = self.maxpool(residual1)
        residual3 = self.maxpool(residual2)
        residual4 = self.maxpool(residual3)

        out4 = self.self_attention4(res4, residual4)

        res3 = self.conv3(res3)
        res3 = self.Fusion3(out4, res3)
        out3 = self.self_attention3(res3, residual3)
        res2 = self.conv2(res2)
        res2 = self.Fusion2(out3, res2)
        out2 = self.self_attention2(res2, residual2)
        res1 = self.conv1(res1)
        res1 = self.Fusion1(out2, res1)
        out1 = self.self_attention1(res1, residual1)


        layer1 = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(out1, dim=1, p=2))))
        layer2 = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(out2, dim=1, p=2))))
        layer3 = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(out3, dim=1, p=2))))
        layer4 =           self.drop2d(self.L2pooling_l4(F.normalize(out4, dim=1, p=2)))
        layers = torch.cat((layer1, layer2, layer3, layer4), dim=1)

        out = torch.flatten(self.avg7(layers),start_dim=1)
        score = self.fc2(out)
        score = self.fc1(score)
        score = self.fc(score)

        return score, CCL