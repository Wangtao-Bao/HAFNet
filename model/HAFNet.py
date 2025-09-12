import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttiton(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttiton, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CBR(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * res

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x_source




def conv_relu_bn(in_channel, out_channel, dirate=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=dirate,
            dilation=dirate,
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )



class Res_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1):
        super(Res_block, self).__init__()
        self.conv_layer = nn.Sequential(
            conv_relu_bn(in_ch, in_ch, 1),
            conv_relu_bn(in_ch, out_ch, 1),
        )

        self.dconv_layer = nn.Sequential(
            conv_relu_bn(in_ch, in_ch, 2),
            conv_relu_bn(in_ch, out_ch, 4),
        )
        self.final_layer = conv_relu_bn(out_ch * 2, out_ch, 1)

        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        conv_out = self.conv_layer(x)
        dconv_out = self.dconv_layer(x)
        out = torch.concat([conv_out,  dconv_out], dim=1)
        out = self.final_layer(out)
        out = self.ca(out)
        out = self.sa(out)
        return out



class HFFE(nn.Module):
    def __init__(self, feature_low_channel, feature_high_channel, out_channel, kernel_size):
        super(HFFE, self).__init__()
        self.conv_block_low = nn.Sequential(
            CBR(feature_low_channel, feature_low_channel // 16, kernel_size),
            nn.Conv2d(feature_low_channel // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv_block_high = nn.Sequential(
            CBR(feature_high_channel, feature_high_channel // 16, kernel_size),
            nn.Conv2d(feature_high_channel // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv1 = CBR(feature_low_channel, out_channel, 1)
        self.conv2 = CBR(feature_high_channel, out_channel, 1)
        self.conv3 = CBR(feature_low_channel + feature_high_channel, out_channel, 1)

        self.Up_to_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.feature_low_sa = SpatialAttention()
        self.feature_high_sa = SpatialAttention()

        self.ca = CoordAttiton(out_channel,out_channel)

        self.conv_final = CBR(out_channel * 2, out_channel, 1)

    def forward(self,x_low, x_high):
        b1, c1, w1, h1 = x_low.size()
        b2, c2, w2, h2 = x_high.size()
        if (w1, h2) != (w2, h2):
            x_high = self.Up_to_2(x_high)

        source_low = x_low
        source_high = x_high

        x_low = self.feature_low_sa(x_low)
        x_high = self.feature_high_sa(x_high)

        x_low_map = self.conv_block_low(x_low)
        x_high_map = self.conv_block_high(x_high)

        x_mix = torch.cat([source_low * x_high_map, source_high * x_low_map], 1)
        x_ca = torch.sigmoid(self.ca(self.conv3(x_mix)))


        x_low_att = x_ca * self.conv1((source_low + x_low))
        x_high_att = x_ca * self.conv2((source_high + x_high))

        out = self.conv_final(torch.cat([x_low_att, x_high_att], 1))

        return out


class HFFD(nn.Module):
    def __init__(self, inchannel_encode, inchannel_hfe, inchannel_decode, out_channel):
        super(HFFD, self).__init__()

        self.conv1 = nn.Conv2d(inchannel_encode, out_channel, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(inchannel_hfe, out_channel, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(inchannel_decode, out_channel, kernel_size=1, stride=1, bias=True)

        self.conv4 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=1, stride=1, bias=True)

        self.layer_conv1 = CBR(out_channel * 2, out_channel, 1)
        self.layer_conv2 = CBR(out_channel * 2, out_channel, 1)
        self.layer_conv3 = CBR(out_channel * 2, out_channel, 1)
        self.layer_conv4 = CBR(out_channel * 2, out_channel * 3, 1)

        # Dilation convolutions
        self.layer_dil1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.layer_dil2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.layer_dil3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=5, dilation=5, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        # Concatenation and output layers
        self.layer_cat = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.layer_out = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x_e, x_hfe ,x_d):

        x_e = self.conv1(x_e)
        x_hfe = self.conv2(x_hfe)
        x_d = self.conv3(x_d)

        x = torch.cat((x_e, x_d),dim=1)

        x1 = self.layer_conv1(x)
        x2 = self.layer_conv2(x)
        x3 = self.layer_conv3(x)
        x4 = self.layer_conv4(x)

        # Apply dilated convolutions
        x_dil3 = self.layer_dil3(x3)
        x_dil2 = self.layer_dil2(x2 + x_dil3)
        x_dil1 = self.layer_dil1(x1 + x_dil2)

        # Concatenate the dilated features
        x_cat = torch.cat((x_dil3, x_dil2, x_dil1), dim=1)

        # Pass through the final layers and output
        out = self.layer_out(x_cat + x4)
        out = self.conv4(torch.cat((out, x_hfe), dim=1))

        return out




class HAFNet(nn.Module):
    def __init__(self,Train=False):
        super().__init__()
        self.Train=Train
        block = Res_block
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.hfe1 = HFFE(param_channels[0], param_channels[1], param_channels[1],1)
        self.hfe2 = HFFE(param_channels[1], param_channels[2], param_channels[2],1)
        self.hfe3 = HFFE(param_channels[2], param_channels[3], param_channels[3],1)
        self.hfe4 = HFFE(param_channels[3], param_channels[4], param_channels[4],1)

        self.hfd4 = HFFD(param_channels[3], param_channels[4], param_channels[4], param_channels[3])
        self.hfd3 = HFFD(param_channels[2], param_channels[3], param_channels[3], param_channels[2])
        self.hfd2 = HFFD(param_channels[1], param_channels[2], param_channels[2], param_channels[1])
        self.hfd1 = HFFD(param_channels[0], param_channels[1], param_channels[1], param_channels[0])



        self.conv_init = nn.Conv2d(1, param_channels[0], 1, 1)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])

        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])

        self.decoder_3 = self._make_layer(param_channels[4], param_channels[3], block,
                                          param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[3], param_channels[2], block,
                                          param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[2], param_channels[1], block,
                                          param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[1], param_channels[0], block)

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)


        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_hfe1 = self.hfe1(x_e0, x_e1)
        x_hfe2 = self.hfe2(x_e1, x_e2)
        x_hfe3 = self.hfe3(x_e2, x_e3)
        x_hfe4 = self.hfe4(x_e3, x_m)

        x_hfd4 = self.hfd4(x_e3,x_hfe4, self.up(x_m))
        x_d3 = self.decoder_3(torch.cat([x_e3, x_hfd4], 1))

        x_hfd3 = self.hfd3(x_e2,x_hfe3, self.up(x_d3))
        x_d2 = self.decoder_2(torch.cat([x_e2, x_hfd3], 1))

        x_hfd2 = self.hfd2(x_e1,x_hfe2, self.up(x_d2))
        x_d1 = self.decoder_1(torch.cat([x_e1, x_hfd2], 1))

        x_hfd1 = self.hfd1(x_e0,x_hfe1, self.up(x_d1))
        x_d0 = self.decoder_0(torch.cat([x_e0, x_hfd1], 1))


        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
        mask1 = F.interpolate(mask1, scale_factor=2, mode='bilinear', align_corners=True)
        mask2 = F.interpolate(mask2, scale_factor=4, mode='bilinear', align_corners=True)
        mask3 = F.interpolate(mask3, scale_factor=8, mode='bilinear', align_corners=True)

        if self.Train:
            return [torch.sigmoid(output),torch.sigmoid(mask0), torch.sigmoid(mask1), torch.sigmoid(mask2),
                    torch.sigmoid(mask3)]
        else:
            return torch.sigmoid(output)


if __name__ == '__main__':

    model = HAFNet(Train=True)
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    flops, params = profile(model, (x,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')

    if len(output)>1:
        print("Output shape:", output[0].shape, output[1].shape, output[2].shape, output[3].shape)
    else:
        print("Output shape:", output.shape)
