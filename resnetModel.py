import torch.nn as nn
import torch


# 残差块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = nn.Conv3d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = nn.Conv3d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self,
                 blocks_num,
                 include_top=False,
                 groups=4,
                 width_per_group=16):
        '''
        :param blocks_num:
        :param include_top: 加载全连接层
        :param groups: 输入通道分成4组同时卷积
        :param width_per_group: 每组16通道
        '''
        super(Resnet, self).__init__()
        self.include_top = include_top
        self.groups = groups
        self.width_per_group = width_per_group
        self.in_channel = 64
        self.pre_due = nn.ZeroPad3d(3)  # 保留边界
        self.conv1 = nn.Conv3d(1, out_channels=self.in_channel, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2)

        self.stage2 = self.__make_layer(64, blocks_num[0])
        self.stage3 = self.__make_layer(128, blocks_num[1], stride=2)
        self.stage4 = self.__make_layer(256, blocks_num[2], stride=2)
        self.stage5 = self.__make_layer(512, blocks_num[3], stride=2)

        self.out1 = nn.Sequential(
            nn.AvgPool3d(2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.out2 = nn.Sequential(
            nn.AvgPool3d(2),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        # self.avgpool = nn.AvgPool3d(2)
        # self.fc = nn.Flatten()
        # self.line1 = nn.Linear(2048, 512)
        # self.act1 = nn.ReLU
        # self.drop = nn.Dropout(0.5)
        # self.line2 = nn.Linear(512, 2)

    def __make_layer(self, first_channel, block_num, stride=1):
        # 第三层单独的卷积深度是一二层的4倍
        if stride != 1 or self.in_channel != first_channel * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channel,
                          first_channel * Bottleneck.expansion,
                          1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm3d(Bottleneck.expansion * first_channel)
            )
        layers = []
        layers.append(Bottleneck(self.in_channel,
                                 first_channel,
                                 downsample=downsample,
                                 groups=self.groups,
                                 width_per_group=self.width_per_group,
                                 stride=stride))
        self.in_channel = first_channel * Bottleneck.expansion
        for _ in range(1, block_num):
            layers.append(Bottleneck(self.in_channel,
                                     first_channel,
                                     groups=self.groups,
                                     width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_due(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        # x = self.out(x)
        x1 = self.out1(x)
        x2 = self.out2(x)

        return x1, x2


def resnet50():
    return Resnet([3, 4, 6, 3], True)


def resnet101():
    return Resnet([3, 4, 23, 3], True)


def resnet151():
    return Resnet([3, 8, 36, 3], True)

# input1 = torch.rand(64, 1, 64, 64, 64)
# model = resnet50()
# a, b = model(input1)
# print(a)
# print(b)
