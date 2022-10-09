import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from axial_block_parallel import HRViTAxialBlock

BN_MOMENTUM = 0.1

from ..lib.bn_helper import  relu_inplace


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats) \
            .permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c, H, W):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super(StageModule, self).__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            if i >= 0:  # 在最高分辨率处还是使用原来的block，下面四个分辨率使用Attention block
                branch = nn.Sequential(
                    HRViTAxialBlock(w, w, H=H // 2 ** i, W=W, heads=8,drop_path = 0.4,attn_drop=0.2, proj_drop=0.2,),
                    # HRViTAxialBlock(w, w, H=H // 2 ** i, W=W),
                    # HRViTAxialBlock(w, w, H=H // 2 ** i, W=W),
                    # HRViTAxialBlock(w, w, H=H // 2 ** i, W=W),
                    # BasicBlock(w, w),
                    # BasicBlock(w, w),
                    # BasicBlock(w, w),
                    # BasicBlock(w, w)
                )
            else:
                branch = nn.Sequential(
                    BasicBlock(w, w),
                    BasicBlock(w, w),
                    BasicBlock(w, w),
                    BasicBlock(w, w)
                )

            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=(2.0 ** (j - i), 1), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=(2, 1), padding=1,
                                          bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=(2, 1), padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused


class HighResolutionNet(nn.Module):
    def __init__(self, input_channel, base_channel: int = 32, in_H: int = 6144, in_W: int = 1, num_joints: int = 17,
                 num_classes: int = 7):
        super(HighResolutionNet, self).__init__()
        self.n_stem = 4
        # Stem
        self.conv1_ = nn.Conv2d(input_channel, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=(2, 1), padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel, H=in_H // self.n_stem, W=in_W)
        )

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=(2, 1), padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # stage2 conv
        self.conv_stage2 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=(1, 3), padding=(0, 0), bias=False),
            nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=(1, 3), padding=(0, 0), bias=False),
            nn.Conv2d(base_channel * 4, base_channel * 4, kernel_size=(1, 3), padding=(0, 0), bias=False)
        )
        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel, H=in_H // self.n_stem, W=in_W // 2 + 1),
            StageModule(input_branches=3, output_branches=3, c=base_channel, H=in_H // self.n_stem, W=in_W // 2 + 1),
            StageModule(input_branches=3, output_branches=3, c=base_channel, H=in_H // self.n_stem, W=in_W // 2 + 1),
            StageModule(input_branches=3, output_branches=3, c=base_channel, H=in_H // self.n_stem, W=in_W // 2 + 1)
        )

        # transition3
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=(2, 1), padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # stage3 conv
        self.conv_stage3 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=(1, 3), padding=(0, 0), bias=False),
            nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=(1, 3), padding=(0, 0), bias=False),
            nn.Conv2d(base_channel * 4, base_channel * 4, kernel_size=(1, 3), padding=(0, 0), bias=False),
            nn.Conv2d(base_channel * 8, base_channel * 8, kernel_size=(1, 3), padding=(0, 0), bias=False)
        )

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel, H=in_H // self.n_stem, W=in_W // 4),
            StageModule(input_branches=4, output_branches=4, c=base_channel, H=in_H // self.n_stem, W=in_W // 4),
            # StageModule(input_branches=4, output_branches=4, c=base_channel),
            # StageModule(input_branches=4, output_branches=4, c=base_channel,H=in_H//4,W=in_W)
        )

        # OCR
        self.last_inp_channels = base_channel * 15  # HRNet融合后的通道数目
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.last_inp_channels, self.last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.last_inp_channels),
            # BatchNorm2d(self.last_inp_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(self.last_inp_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.ocr_mid_channels = 512
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(self.last_inp_channels, self.ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            # BatchNorm2d(self.ocr_mid_channels),
            nn.BatchNorm2d(self.ocr_mid_channels),
            nn.ReLU(inplace=relu_inplace),
        )

        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_key_channels = 256
        self.ocr_distri_head = SpatialOCR_Module(in_channels=self.ocr_mid_channels,
                                                 key_channels=self.ocr_key_channels,
                                                 out_channels=self.ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        # Final layer

        self.cls_head = nn.Conv2d(
            self.ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        # self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)

    def forward(self, x):
        # print(self.)
        B, C, H, W = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print('layer1',x.shape)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list

        # for i,n in enumerate(x):
        #     print("stage1,layer{}".format(i+1),n.shape)

        x = self.stage2(x)

        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only
        # print('-'*19)

        # for i,n in enumerate(x):
        #     print("stage2,layer{}".format(i+1),n.shape)

        x = [
            self.conv_stage2[0](x[0]),
            self.conv_stage2[1](x[1]),
            self.conv_stage2[2](x[-1])
        ]  # 经过conv将横向分辨率缩小

        # for i,n in enumerate(x):
        #     print("stage2,layer{}".format(i+1),n.shape)

        x = self.stage3(x)

        # for i in range(len(x)):
        #     print('stage:x{}'.format(i),x[i].shape)

        # print('-' * 19)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only

        # for i,n in enumerate(x):
        #     print("stage3,layer{}".format(i+1),n.shape)

        x = [
            self.conv_stage3[0](x[0]),
            self.conv_stage3[1](x[1]),
            self.conv_stage3[2](x[2]),
            self.conv_stage3[3](x[-1])
        ]  # 经过conv将横向分辨率缩小

        # for i,n in enumerate(x):
        #     print("stage3,layer{}".format(i+1),n.shape)

        x = self.stage4(x)
        # for i,n in enumerate(x):
        #     print("stage4,layer{}".format(i+1),n.shape)
        # print('-' * 19)
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x0 = F.interpolate(x[0], size=(H, W // 4),
                           mode='bilinear', align_corners=True)
        x1 = F.interpolate(x[1], size=(H, W // 4),
                           mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(H, W // 4),
                           mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(H, W // 4),
                           mode='bilinear', align_corners=True)

        # print(x[0].shape,x1.shape,x2.shape,x3.shape)
        feats = torch.cat([x0, x1, x2, x3], 1)
        # print('feats',feats.shape)
        out_aux_seg = []

        # ocr 粗分割结果 soft object regions
        out_aux = self.aux_head(feats)
        # compute contrast feature  pixel representations
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        return out_aux_seg

        # hr1 = x[0]
        # hr2 = x[1]
        # hr3 = x[2]
        # hr4 = x[3]
        # for x_ in x:
        #     print(x_.shape)
        #
        # x = self.final_layer(x[0])
        #
        # return x


def hr_ocr_w48(in_channel, out_channel):
    return HighResolutionNet(base_channel=48, input_channel=in_channel, num_classes=out_channel), hr_ocr_w48.__name__


def hr_ocr_w32(in_channel, out_channel):
    return HighResolutionNet(base_channel=32, input_channel=in_channel, num_classes=out_channel), hr_ocr_w32.__name__


def hr_ocr_w18(in_channel, out_channel):
    return HighResolutionNet(base_channel=18, input_channel=in_channel, num_classes=out_channel), hr_ocr_w18.__name__


def hr_ocr_axial_w(base_channel, in_channel, out_channel, H, W):
    return HighResolutionNet(base_channel=base_channel, input_channel=in_channel,
                             num_classes=out_channel,
                             in_H=H, in_W=W), hr_ocr_axial_w.__name__


if __name__ == '__main__':
    device = torch.device('cuda')
    # device = torch.device('cpu')
    batch_size = 2
    in_channel, out_channel = 1, 7
    base_channel = 24
    H, W = 3072, 5
    # print(W//2)
    used_model, name = hr_ocr_axial_w(base_channel, in_channel, out_channel, H, W)
    used_model = used_model.to(device)
    print(name)
    input = torch.rand(batch_size, in_channel, H, W).to(device)
    # model = HighResolutionNet()
    # summary(used_model,(1,128,5))
    output, _ = used_model(input)
    # print(model)
    print(output.shape)
    print(_.shape)
    # summary(used_model, input_size=(5, 128, 5))