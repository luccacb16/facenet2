import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = ConvBlock(256, 32, kernel_size=1)
        self.branch1 = nn.Sequential(
            ConvBlock(256, 32, kernel_size=1),
            ConvBlock(32, 32, kernel_size=3, padding=1)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(256, 32, kernel_size=1),
            ConvBlock(32, 32, kernel_size=3, padding=1),
            ConvBlock(32, 32, kernel_size=3, padding=1)
        )
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = ConvBlock(896, 128, kernel_size=1)
        self.branch1 = nn.Sequential(
            ConvBlock(896, 128, kernel_size=1),
            ConvBlock(128, 128, kernel_size=(1, 7), padding=(0, 3)),
            ConvBlock(128, 128, kernel_size=(7, 1), padding=(3, 0))
        )
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Block8(nn.Module):
    def __init__(self, scale=1.0, activation_fn=True):
        super(Block8, self).__init__()
        self.scale = scale
        self.activation_fn = activation_fn
        self.branch0 = ConvBlock(1792, 192, kernel_size=1)
        self.branch1 = nn.Sequential(
            ConvBlock(1792, 192, kernel_size=1),
            ConvBlock(192, 192, kernel_size=(1, 3), padding=(0, 1)),
            ConvBlock(192, 192, kernel_size=(3, 1), padding=(1, 0))
        )
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if self.activation_fn:
            out = self.relu(out)
        return out

class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = ConvBlock(256, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            ConvBlock(256, 192, kernel_size=1),
            ConvBlock(192, 192, kernel_size=3, padding=1),
            ConvBlock(192, 256, kernel_size=3, stride=2)
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Mixed_7a(nn.Module):
    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(
            ConvBlock(896, 256, kernel_size=1),
            ConvBlock(256, 384, kernel_size=3, stride=2)
        )
        self.branch1 = nn.Sequential(
            ConvBlock(896, 256, kernel_size=1),
            ConvBlock(256, 256, kernel_size=3, stride=2)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(896, 256, kernel_size=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=2)
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class InceptionResnetV1(nn.Module):
    def __init__(self, emb_size: int = 64):
        super(InceptionResnetV1, self).__init__()
        self.conv2d_1a = ConvBlock(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = ConvBlock(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = ConvBlock(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = ConvBlock(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = ConvBlock(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(*[Block35(scale=0.17) for _ in range(5)])
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(*[Block17(scale=0.10) for _ in range(10)])
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(*[Block8(scale=0.20) for _ in range(5)])
        self.block8 = Block8(activation_fn=False)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.6)
        self.last_linear = nn.Linear(1792, emb_size, bias=False)
        self.last_bn = nn.BatchNorm1d(emb_size, eps=0.001, momentum=0.1, affine=True)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        
        return F.normalize(x, p=2, dim=1)