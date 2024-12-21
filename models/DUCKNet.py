import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_initializer = 'he_uniform'
interpolation = "nearest"


class SeparatedConv2DBlock(nn.Module):
    def __init__(self, in_channels, size=3, down=False):
        super(SeparatedConv2DBlock, self).__init__()
        self.size = size
        half_channels = int(in_channels // 2) if down else in_channels

        self.conv1 = nn.Conv2d(in_channels, half_channels, kernel_size=(1, size), stride=1).cuda()
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(half_channels).cuda()

        self.conv2 = nn.Conv2d(half_channels, half_channels, kernel_size=(size, 1), stride=1).cuda()
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(half_channels).cuda()

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        if self.size % 2 == 0:
            # pad1 = (self.size // 2, self.size // 2, 0, 0)
            # pad2 = (0, 0, self.size // 2, self.size // 2)
            pad1 = ((self.size - 1) // 2, self.size // 2, 0, 0)
            pad2 = (0, 0, (self.size - 1) // 2, self.size // 2)

        else:
            pad1 = (self.size // 2, self.size // 2, 0, 0)
            pad2 = (0, 0, self.size // 2, self.size // 2)

        # pad1 = (self.size // 2, self.size // 2, 0, 0)
        x = torch.nn.functional.pad(x, pad1, mode='constant', value=0)
        x = self.relu1(self.conv1(x))
        x = self.bn1(x)

        # pad2 = (0, 0, self.size // 2, self.size // 2)
        x = torch.nn.functional.pad(x, pad2, mode='constant', value=0)
        x = self.relu2(self.conv2(x))
        x = self.bn2(x)
        return x


class WideScopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, size=3, padding='same', down=False):
        super(WideScopeConv2DBlock, self).__init__()
        half_channels = int(in_channels // 2) if down else in_channels

        self.conv1 = nn.Conv2d(in_channels, half_channels, kernel_size=(size, size), padding=padding, dilation=1).cuda()
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(half_channels).cuda()

        self.conv2 = nn.Conv2d(half_channels, half_channels, kernel_size=(size, size), padding=padding, dilation=2).cuda()
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(half_channels).cuda()

        self.conv3 = nn.Conv2d(half_channels, half_channels, kernel_size=(size, size), padding=padding, dilation=3).cuda()
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(half_channels).cuda()

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.bn1(x)

        x = self.relu2(self.conv2(x))
        x = self.bn2(x)

        x = self.relu3(self.conv3(x))
        x = self.bn3(x)

        return x


class MidScopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, size=3, padding='same', down=False):
        super(MidScopeConv2DBlock, self).__init__()
        half_channels = int(in_channels // 2) if down else in_channels

        self.conv1 = nn.Conv2d(in_channels, half_channels, kernel_size=(size, size), padding=padding, dilation=1).cuda()
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(half_channels).cuda()

        self.conv2 = nn.Conv2d(half_channels, half_channels, kernel_size=(size, size), padding=padding, dilation=2).cuda()
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(half_channels).cuda()

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.bn1(x)

        x = self.relu2(self.conv2(x))
        x = self.bn2(x)

        return x


class ResNetConv2DBlock(nn.Module):
    def __init__(self, in_channels, dilation=1, padding='same', down=False):
        super(ResNetConv2DBlock, self).__init__()
        half_channels = int(in_channels // 2) if down else in_channels
        # half_channels = in_channels
        self.in_channels = in_channels
        self.half_channels = half_channels

        self.conv_ = nn.Conv2d(in_channels, half_channels, kernel_size=(1, 1), padding=0, dilation=dilation).cuda()
        self.relu_ = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, half_channels, kernel_size=(3, 3), padding=1, dilation=dilation).cuda()
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(half_channels).cuda()

        self.conv2 = nn.Conv2d(half_channels, half_channels, kernel_size=(3, 3), padding=1, dilation=dilation).cuda()
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(half_channels).cuda()

        self.bn3 = nn.BatchNorm2d(half_channels).cuda()

        nn.init.kaiming_uniform_(self.conv_.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        x1 = self.relu_(self.conv_(x))

        x = self.relu1(self.conv1(x))
        x = self.bn1(x)

        x = self.relu2(self.conv2(x))
        x = self.bn2(x)

        x = self.bn3(x + x1)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, size=3, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(size, size), padding=padding)
        self.relu1 = nn.ReLU()

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        return x


class DuckV2Conv2DBlock(nn.Module):
    def __init__(self, in_channels, size=3, down=False):
        super(DuckV2Conv2DBlock, self).__init__()
        self.in_channels = in_channels

        self.bn1 = nn.BatchNorm2d(in_channels).cuda()
        self.widescope = WideScopeConv2DBlock(in_channels, size, down=down)
        self.midscope = MidScopeConv2DBlock(in_channels, size, down=down)
        self.conv2d_block1 = ConvBlock2D(in_channels, 'resnet', repeat=1, down=down)

        if down:
            self.conv2d_block2 = nn.Sequential(
                ConvBlock2D(in_channels, 'resnet', repeat=1, down=down),
                ConvBlock2D(in_channels // 2, 'resnet', repeat=1)
            )
            self.conv2d_block3 = nn.Sequential(
                ConvBlock2D(in_channels, 'resnet', repeat=1, down=down),
                ConvBlock2D(in_channels // 2, 'resnet', repeat=1),
                ConvBlock2D(in_channels // 2, 'resnet', repeat=1)
            )
        else:
            self.conv2d_block2 = ConvBlock2D(in_channels, 'resnet', repeat=2, down=down)
            self.conv2d_block3 = ConvBlock2D(in_channels, 'resnet', repeat=3, down=down)

        self.separated = SeparatedConv2DBlock(in_channels, size=6, down=down)
        self.bn2 = nn.BatchNorm2d(in_channels).cuda() if not down else nn.BatchNorm2d(in_channels // 2).cuda()

    def forward(self, x):
        x = x.cuda()
        x = self.bn1(x)
        x1 = self.widescope(x)
        x2 = self.midscope(x)
        x3 = self.conv2d_block1(x)
        x4 = self.conv2d_block2(x)
        x5 = self.conv2d_block3(x)
        x6 = self.separated(x)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.bn2(x)

        return x


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, block_type, repeat=1, dilation=1, size=3, padding='same', down=False):
        super(ConvBlock2D, self).__init__()
        self.repeat = repeat

        self.layers = []
        for i in range(repeat):
            if block_type == 'separated':
                layer = SeparatedConv2DBlock(in_channels, size)
            elif block_type == 'duckv2':
                layer = DuckV2Conv2DBlock(in_channels, size, down=down)
            elif block_type == 'widescope':
                layer = WideScopeConv2DBlock(in_channels)
            elif block_type == 'midscope':
                layer = MidScopeConv2DBlock(in_channels)
            elif block_type == 'resnet':
                layer = ResNetConv2DBlock(in_channels, dilation, down=down)
            elif block_type == 'conv':
                layer = ConvBlock(in_channels, size, padding=padding)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DuckNet(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(DuckNet, self).__init__()
        self.in_channels = in_channels

        self.p1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2, padding=0)
        self.p2 = nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=2, stride=2, padding=0)
        self.p3 = nn.Conv2d(in_channels * 4, in_channels * 8, kernel_size=2, stride=2, padding=0)
        self.p4 = nn.Conv2d(in_channels * 8, in_channels * 16, kernel_size=2, stride=2, padding=0)
        self.p5 = nn.Conv2d(in_channels * 16, in_channels * 32, kernel_size=2, stride=2, padding=0)

        self.t0 = ConvBlock2D(in_channels, 'duckv2', repeat=1)

        self.l1i = nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2, padding=0)
        self.t1 = ConvBlock2D(in_channels * 2, 'duckv2', repeat=1)

        self.l2i = nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=2, stride=2, padding=0)
        self.t2 = ConvBlock2D(in_channels * 4, 'duckv2', repeat=1)

        self.l3i = nn.Conv2d(in_channels * 4, in_channels * 8, kernel_size=2, stride=2, padding=0)
        self.t3 = ConvBlock2D(in_channels * 8, 'duckv2', repeat=1)

        self.l4i = nn.Conv2d(in_channels * 8, in_channels * 16, kernel_size=2, stride=2, padding=0)
        self.t4 = ConvBlock2D(in_channels * 16, 'duckv2', repeat=1)

        self.l5i = nn.Conv2d(in_channels * 16, in_channels * 32, kernel_size=2, stride=2, padding=0)
        self.t51 = ConvBlock2D(in_channels * 32, 'resnet', repeat=2)
        self.t53 = nn.Sequential(
            ConvBlock2D(in_channels * 32, 'resnet', repeat=1, down=True),
            ConvBlock2D(in_channels * 16, 'resnet', repeat=1)
        )
        self.q4 = ConvBlock2D(in_channels * 16, 'duckv2', repeat=1, down=True)

        self.q3 = ConvBlock2D(in_channels * 8, 'duckv2', repeat=1, down=True)

        self.q6 = ConvBlock2D(in_channels * 4, 'duckv2', repeat=1, down=True)

        self.q1 = ConvBlock2D(in_channels * 2, 'duckv2', repeat=1, down=True)

        self.z1 = ConvBlock2D(in_channels, 'duckv2', repeat=1)

        self.out = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        t0 = self.t0(x)

        l1i = self.l1i(t0)
        s1 = p1 + l1i
        t1 = self.t1(s1)

        l2i = self.l2i(t1)
        s2 = p2 + l2i
        t2 = self.t2(s2)

        l3i = self.l3i(t2)
        s3 = p3 + l3i
        t3 = self.t3(s3)

        l4i = self.l4i(t3)
        s4 = p4 + l4i
        t4 = self.t4(s4)

        l5i = self.l5i(t4)
        s5 = p5 + l5i
        t51 = self.t51(s5)
        t53 = self.t53(t51)

        l5o = F.interpolate(t53, scale_factor=(2, 2), mode=interpolation)
        c4 = l5o + t4
        q4 = self.q4(c4)

        l4o = F.interpolate(q4, scale_factor=(2, 2), mode=interpolation)
        c3 = l4o + t3
        q3 = self.q3(c3)

        l3o = F.interpolate(q3, scale_factor=(2, 2), mode=interpolation)
        c2 = l3o + t2
        q6 = self.q6(c2)

        l2o = F.interpolate(q6, scale_factor=(2, 2), mode=interpolation)
        c1 = l2o + t1
        q1 = self.q1(c1)

        l1o = F.interpolate(q1, scale_factor=(2, 2), mode=interpolation)
        c0 = l1o + t0
        z1 = self.z1(c0)
        out = self.out(z1)
        out = torch.relu(out)
        # out = F.sigmoid(self.out(z1))
        out = torch.exp(-out) + out - 1

        return out


if __name__ == "__main__":
    model = DuckNet(in_channels=3, num_classes=1)
    print(model)
    x = torch.ones(1, 3, 256, 256)
    y = model(x)
    print(y.shape)

    dummy_input = torch.ones(1, 3, 256, 256)

    import torchinfo

    torchinfo.summary(model, input_data=dummy_input)