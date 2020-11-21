from __future__ import absolute_import, print_function
import torch
from torch import nn

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

def permute_dim(x):
    b, c, t, h, w = x.size()
    tt = x.permute(0, 2, 1, 3, 4)
    tensor = tt.contiguous().view(b * t, c, h, w)
    return tensor, t

class ITAE_encoder(nn.Module):
    def __init__(self, chnum_in, one_path=False):
        super(ITAE_encoder, self).__init__()
        self.chnum_in = chnum_in
        feature_num = 96
        feature_num_2 = 128
        feature_num_3 = 256
        beta = 8
        self.one_path = one_path

        # static path
        self.conv1 = nn.Conv3d(self.chnum_in, feature_num, (1,3,3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(feature_num)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        if not one_path:
            self.conv2 = nn.Conv3d((feature_num+ int(feature_num/beta)*2), feature_num_2, (1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False)
        else:
            self.conv2 = nn.Conv3d((feature_num ), feature_num_2, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(feature_num_2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        if not one_path:
            self.conv3 = nn.Conv3d((feature_num_2+int(feature_num_2/beta)*2), feature_num_3, (3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)
        else:
            self.conv3 = nn.Conv3d((feature_num_2 ), feature_num_3, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(feature_num_3)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        if not one_path:
            self.conv4 = nn.Conv3d((feature_num_3+int(feature_num_3/beta)*2), feature_num_3, (3,3,3),  padding=(1, 1, 1), bias=False)
        else:
            self.conv4 = nn.Conv3d((feature_num_3 ), feature_num_3, (3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn4= nn.BatchNorm3d(feature_num_3)
        self.act4 =nn.LeakyReLU(0.2, inplace=True)

        self.CP_static = ChannelPool()

        if not one_path:
            # dynamic path
            self.conv1_f = nn.Conv3d(self.chnum_in, int(feature_num / beta), (5, 3, 3), stride=(1, 2, 2),
                                     padding=(2, 1, 1), bias=False)
            self.bn1_f = nn.BatchNorm3d(int(feature_num / beta))
            self.act1_f = nn.LeakyReLU(0.2, inplace=True)
            self.lateral_1 = nn.Conv3d(int(feature_num / beta), int(feature_num / beta) * 2, (5, 1, 1),
                                       stride=(int(beta / 2), 1, 1), padding=(2, 0, 0), bias=False)

            self.conv2_f = nn.Conv3d(int(feature_num / beta), int(feature_num_2 / beta), (3, 3, 3), stride=(1, 2, 2),
                                     padding=(1, 1, 1), bias=False)
            self.bn2_f = nn.BatchNorm3d(int(feature_num_2 / beta))
            self.act2_f = nn.LeakyReLU(0.2, inplace=True)
            self.lateral_2 = nn.Conv3d(int(feature_num_2 / beta), int(feature_num_2 / beta) * 2, (5, 1, 1),
                                       stride=(int(beta / 2), 1, 1), padding=(2, 0, 0), bias=False)

            self.conv3_f = nn.Conv3d(int(feature_num_2 / beta), int(feature_num_3 / beta), (3, 3, 3), stride=(1, 2, 2),
                                     padding=(1, 1, 1), bias=False)
            self.bn3_f = nn.BatchNorm3d(int(feature_num_3 / beta))
            self.act3_f = nn.LeakyReLU(0.2, inplace=True)
            self.lateral_3 = nn.Conv3d(int(feature_num_3 / beta), int(feature_num_3 / beta) * 2, (5, 1, 1),
                                       stride=(int(beta / 2), 1, 1), padding=(2, 0, 0), bias=False)

            self.conv4_f = nn.Conv3d(int(feature_num_3 / beta), int(feature_num_3 / beta), (3, 3, 3),
                                     padding=(1, 1, 1), bias=False)
            self.bn4_f = nn.BatchNorm3d(int(feature_num_3 / beta))
            self.act4_f = nn.LeakyReLU(0.2, inplace=True)

            self.CP_dynamic = ChannelPool()

            self.lateral_final = nn.Conv3d(int(feature_num_3 / beta), int(feature_num_3 / beta) * 2, (5, 1, 1),
                                           stride=(int(beta / 2), 1, 1), padding=(2, 0, 0), bias=False)

    def forward(self, input):
        if not self.one_path:
            dynamic, lateral = self.DynamicPath(input[:, :, ::1, :, :])
        else:
            lateral = None
        static, flow_static = self.StaticPath(input[:, :, ::4, :, :], lateral)

        if not self.one_path:
            dynamic = self.lateral_final(dynamic)
            output = torch.cat([static, dynamic], dim=1)

            flow_dynamic = dynamic.clone()
            flow_dynamic = self.CP_dynamic(flow_dynamic)
            flow_dynamic, _ = permute_dim(flow_dynamic)
        else:
            output = static
            flow_dynamic = static

        return output, (flow_static, flow_dynamic)

    def StaticPath(self, input, lateral = None):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        if lateral is not None:
            x = torch.cat([x, lateral[0]], dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if lateral is not None:
            x = torch.cat([x, lateral[1]], dim=1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        if lateral is not None:
            x = torch.cat([x, lateral[2]], dim=1)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        flow_z = self.CP_static(x)
        flow_z, _ = permute_dim(flow_z)

        return x, flow_z

    def DynamicPath(self, input):
        lateral = []
        x = self.conv1_f(input)
        x = self.bn1_f(x)
        x = self.act1_f(x)
        lateral1 = self.lateral_1(x)
        lateral.append(lateral1)


        x = self.conv2_f(x)
        x = self.bn2_f(x)
        x = self.act2_f(x)
        lateral2 = self.lateral_2(x)
        lateral.append(lateral2)

        x = self.conv3_f(x)
        x = self.bn3_f(x)
        x = self.act3_f(x)
        lateral3 = self.lateral_3(x)
        lateral.append(lateral3)

        x = self.conv4_f(x)
        x = self.bn4_f(x)
        x = self.act4_f(x)

        return x, lateral

class ITAE_decoder(nn.Module):
    def __init__(self, chnum_in, one_path = False):
        super(ITAE_decoder, self).__init__()
        self.chnum_in = chnum_in
        beta = 8
        feature_num = 96
        feature_num_2 = 128
        if not one_path:
            feature_num_3 = 256 + int(256/beta*2)
        else:
            feature_num_3 = 256

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_3, feature_num_3, (3, 3, 3), stride=(1,1,1),  padding=(1, 1, 1), output_padding=(0,0, 0)),
            nn.BatchNorm3d(feature_num_3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_3, feature_num_2, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, feature_num, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, self.chnum_in, (3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1))
        )

    def forward(self, x):
        out = self.decoder(x)
        return out

class ITAE_encoder_x4(nn.Module):
    def __init__(self, chnum_in, one_path=False):
        super(ITAE_encoder_x4, self).__init__()
        self.chnum_in = chnum_in
        feature_num = 96
        feature_num_2 = 128
        feature_num_3 = 256
        beta = 8

        self.one_path = one_path
        # static path
        self.conv1 = nn.Conv3d(self.chnum_in, feature_num, (1,3,3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(feature_num)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        if not one_path:
            self.conv2 = nn.Conv3d((feature_num+ int(feature_num/beta)*2), feature_num_2, (1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False)
        else:
            self.conv2 = nn.Conv3d((feature_num ), feature_num_2, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(feature_num_2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        if not one_path:
            self.conv3 = nn.Conv3d((feature_num_2+int(feature_num_2/beta)*2), feature_num_3, (3,3,3), padding=(1,1,1), bias=False)
        else:
            self.conv3 = nn.Conv3d((feature_num_2 ), feature_num_3, (3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(feature_num_3)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        if not one_path:
            self.conv4 = nn.Conv3d((feature_num_3+int(feature_num_3/beta)*2), feature_num_3, (3,3,3),  padding=(1, 1, 1), bias=False)
        else:
            self.conv4 = nn.Conv3d((feature_num_3 ), feature_num_3, (3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn4= nn.BatchNorm3d(feature_num_3)
        self.act4 =nn.LeakyReLU(0.2, inplace=True)

        self.CP_static = ChannelPool()

        if not one_path:
            # dynamic path
            self.conv1_f = nn.Conv3d(self.chnum_in, int(feature_num / beta), (5, 3, 3), stride=(1, 2, 2),
                                     padding=(2, 1, 1), bias=False)
            self.bn1_f = nn.BatchNorm3d(int(feature_num / beta))
            self.act1_f = nn.LeakyReLU(0.2, inplace=True)
            self.lateral_1 = nn.Conv3d(int(feature_num / beta), int(feature_num / beta) * 2, (5, 1, 1),
                                       stride=(int(beta / 2), 1, 1), padding=(2, 0, 0), bias=False)

            self.conv2_f = nn.Conv3d(int(feature_num / beta), int(feature_num_2 / beta), (3, 3, 3), stride=(1, 2, 2),
                                     padding=(1, 1, 1), bias=False)
            self.bn2_f = nn.BatchNorm3d(int(feature_num_2 / beta))
            self.act2_f = nn.LeakyReLU(0.2, inplace=True)
            self.lateral_2 = nn.Conv3d(int(feature_num_2 / beta), int(feature_num_2 / beta) * 2, (5, 1, 1),
                                       stride=(int(beta / 2), 1, 1), padding=(2, 0, 0), bias=False)

            self.conv3_f = nn.Conv3d(int(feature_num_2 / beta), int(feature_num_3 / beta), (3, 3, 3),
                                     padding=(1, 1, 1), bias=False)
            self.bn3_f = nn.BatchNorm3d(int(feature_num_3 / beta))
            self.act3_f = nn.LeakyReLU(0.2, inplace=True)
            self.lateral_3 = nn.Conv3d(int(feature_num_3 / beta), int(feature_num_3 / beta) * 2, (5, 1, 1),
                                       stride=(int(beta / 2), 1, 1), padding=(2, 0, 0), bias=False)

            self.conv4_f = nn.Conv3d(int(feature_num_3 / beta), int(feature_num_3 / beta), (3, 3, 3),
                                     padding=(1, 1, 1), bias=False)
            self.bn4_f = nn.BatchNorm3d(int(feature_num_3 / beta))
            self.act4_f = nn.LeakyReLU(0.2, inplace=True)

            self.CP_dynamic = ChannelPool()

            self.lateral_final = nn.Conv3d(int(feature_num_3 / beta), int(feature_num_3 / beta) * 2, (5, 1, 1),
                                           stride=(int(beta / 2), 1, 1), padding=(2, 0, 0), bias=False)

    def forward(self, input):
        if not self.one_path:
            dynamic, lateral = self.DynamicPath(input[:, :, ::1, :, :])
        else:
            lateral = None
        static, flow_static = self.StaticPath(input[:, :, ::4, :, :], lateral)

        if not self.one_path:
            dynamic = self.lateral_final(dynamic)
            output = torch.cat([static, dynamic], dim=1)

            flow_dynamic = dynamic.clone()
            flow_dynamic = self.CP_dynamic(flow_dynamic)
            flow_dynamic, _ = permute_dim(flow_dynamic)
        else:
            output = static
            flow_dynamic = flow_static

        return output, (flow_static, flow_dynamic)

    def StaticPath(self, input, lateral = None):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        if lateral is not None:
            x = torch.cat([x, lateral[0]], dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if lateral is not None:
            x = torch.cat([x, lateral[1]], dim=1)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        if lateral is not None:
            x = torch.cat([x, lateral[2]], dim=1)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        flow_z = self.CP_static(x)
        flow_z, _ = permute_dim(flow_z)

        return x, flow_z

    def DynamicPath(self, input):
        lateral = []
        x = self.conv1_f(input)
        x = self.bn1_f(x)
        x = self.act1_f(x)
        lateral1 = self.lateral_1(x)
        lateral.append(lateral1)


        x = self.conv2_f(x)
        x = self.bn2_f(x)
        x = self.act2_f(x)
        lateral2 = self.lateral_2(x)
        lateral.append(lateral2)

        x = self.conv3_f(x)
        x = self.bn3_f(x)
        x = self.act3_f(x)
        lateral3 = self.lateral_3(x)
        lateral.append(lateral3)

        x = self.conv4_f(x)
        x = self.bn4_f(x)
        x = self.act4_f(x)

        return x, lateral

class ITAE_decoder_x4(nn.Module):
    def __init__(self, chnum_in, one_path = False):
        super(ITAE_decoder_x4, self).__init__()
        self.chnum_in = chnum_in
        beta = 8
        feature_num = 96
        feature_num_2 = 128
        if not one_path:
            feature_num_3 = 256 + int(256/beta*2)
        else:
            feature_num_3 = 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_3, feature_num_3, (3, 3, 3), stride=(1,1,1),  padding=(1, 1, 1), output_padding=(0,0, 0)),
            nn.BatchNorm3d(feature_num_3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_3, feature_num_2, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, feature_num, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, self.chnum_in, (3,3,3), stride=(1,1, 1), padding=(1,1,1), output_padding=(0,0, 0))
        )

    def forward(self, x):
        out = self.decoder(x)
        return out