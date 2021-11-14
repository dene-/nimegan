import json
from collections import OrderedDict
from math import exp

from Common import *


# +++++++++++++++++++++++++++++++++++++
#           FP16 Training      
# -------------------------------------
#  Modified from Nvidia/Apex
# https://github.com/NVIDIA/apex/blob/master/apex/fp16_utils/fp16util.py

class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        if input.is_cuda:
            return input.half()
        else:  # PyTorch 1.0 doesn't support fp16 in CPU
            return input.float()


def BN_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))

# +++++++++++++++++++++++++++++++++++++
#           CARN      
# -------------------------------------

class CARN_Block(BaseModule):
    def __init__(self, channels, kernel_size=3, padding=1, dilation=1,
                 groups=1, activation=nn.SELU(), repeat=3,
                 SEBlock=False, conv=nn.Conv2d,
                 single_conv_size=1, single_conv_group=1):
        super(CARN_Block, self).__init__()
        m = []
        for i in range(repeat):
            m.append(ResidualFixBlock(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                      groups=groups, activation=activation, conv=conv))
            if SEBlock:
                m.append(SpatialChannelSqueezeExcitation(channels, reduction=channels))
        self.blocks = nn.Sequential(*m)
        self.singles = nn.Sequential(
            *[ConvBlock(channels * (i + 2), channels, kernel_size=single_conv_size,
                        padding=(single_conv_size - 1) // 2, groups=single_conv_group,
                        activation=activation, conv=conv)
              for i in range(repeat)])

    def forward(self, x):
        c0 = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)

        return x


class CARN(BaseModule):
    # Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network
    # https://github.com/nmhkahn/CARN-pytorch
    def __init__(self,
                 color_channels=3,
                 mid_channels=64,
                 scale=2,
                 activation=nn.SELU(),
                 num_blocks=3,
                 conv=nn.Conv2d):
        super(CARN, self).__init__()

        self.color_channels = color_channels
        self.mid_channels = mid_channels
        self.scale = scale

        self.entry_block = ConvBlock(color_channels, mid_channels, kernel_size=3, padding=1, activation=activation,
                                     conv=conv)
        self.blocks = nn.Sequential(
            *[CARN_Block(mid_channels, kernel_size=3, padding=1, activation=activation, conv=conv,
                         single_conv_size=1, single_conv_group=1)
              for _ in range(num_blocks)])
        self.singles = nn.Sequential(
            *[ConvBlock(mid_channels * (i + 2), mid_channels, kernel_size=1, padding=0,
                        activation=activation, conv=conv)
              for i in range(num_blocks)])

        self.upsampler = UpSampleBlock(mid_channels, scale=scale, activation=activation, conv=conv)
        self.exit_conv = conv(mid_channels, color_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry_block(x)
        c0 = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)
        x = self.upsampler(x)
        out = self.exit_conv(x)
        return out


class CARN_V2(CARN):
    def __init__(self, color_channels=3, mid_channels=64,
                 scale=2, activation=nn.LeakyReLU(0.1),
                 SEBlock=True, conv=nn.Conv2d,
                 atrous=(1, 1, 1), repeat_blocks=3,
                 single_conv_size=3, single_conv_group=1):
        super(CARN_V2, self).__init__(color_channels=color_channels, mid_channels=mid_channels, scale=scale,
                                      activation=activation, conv=conv)

        num_blocks = len(atrous)
        m = []
        for i in range(num_blocks):
            m.append(CARN_Block(mid_channels, kernel_size=3, padding=1, dilation=1,
                                activation=activation, SEBlock=SEBlock, conv=conv, repeat=repeat_blocks,
                                single_conv_size=single_conv_size, single_conv_group=single_conv_group))

        self.blocks = nn.Sequential(*m)

        self.singles = nn.Sequential(
            *[ConvBlock(mid_channels * (i + 2), mid_channels, kernel_size=single_conv_size,
                        padding=(single_conv_size - 1) // 2, groups=single_conv_group,
                        activation=activation, conv=conv)
              for i in range(num_blocks)])

    def forward(self, x):
        x = self.entry_block(x)
        c0 = x
        res = x
        for block, single in zip(self.blocks, self.singles):
            b = block(x)
            c0 = c = torch.cat([c0, b], dim=1)
            x = single(c)
        x = x + res
        x = self.upsampler(x)
        out = self.exit_conv(x)
        return out


# +++++++++++++++++++++++++++++++++++++
#           original Waifu2x model
# -------------------------------------


class UpConv_7(BaseModule):
    # https://github.com/nagadomi/waifu2x/blob/3c46906cb78895dbd5a25c3705994a1b2e873199/lib/srcnn.lua#L311
    def __init__(self):
        super(UpConv_7, self).__init__()
        self.act_fn = nn.LeakyReLU(0.1, inplace=False)
        self.offset = 7  # because of 0 padding
        from torch.nn import ZeroPad2d
        self.pad = ZeroPad2d(self.offset)
        m = [nn.Conv2d(3, 16, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(16, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 256, 3, 1, 0),
             self.act_fn,
             # in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=
             nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=3, bias=False)
             ]
        self.Sequential = nn.Sequential(*m)

    def load_pre_train_weights(self, json_file):
        with open(json_file) as f:
            weights = json.load(f)
        box = []
        for i in weights:
            box.append(i['weight'])
            box.append(i['bias'])
        own_state = self.state_dict()
        for index, (name, param) in enumerate(own_state.items()):
            own_state[name].copy_(torch.FloatTensor(box[index]))

    def forward(self, x):
        x = self.pad(x)
        return self.Sequential.forward(x)



class Vgg_7(UpConv_7):
    def __init__(self):
        super(Vgg_7, self).__init__()
        self.act_fn = nn.LeakyReLU(0.1, inplace=False)
        self.offset = 7
        m = [nn.Conv2d(3, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 3, 3, 1, 0)
             ]
        self.Sequential = nn.Sequential(*m)