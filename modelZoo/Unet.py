import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def convrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding,  groups=1, bias=False, dilation=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )


class viewTransformer(nn.Module):
    def __init__(self, in_channel):
    # def __init__(self):
        super(viewTransformer,self).__init__()
        # self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)

        self.layer0 = convrelu(in_channels=in_channel, out_channels=64, kernel=3, stride=2, padding=1)
        self.layer0_1x1 = convrelu(in_channels=64, out_channels=64, kernel=1, stride=1, padding=0)
        # self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)

        self.layer1 = convrelu(in_channels=64, out_channels=64, kernel=3, stride=2, padding=1)
        self.layer1_1x1 = convrelu(in_channels=64, out_channels=64, kernel=1, stride=1, padding=0)

        # self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2 = convrelu(in_channels=64, out_channels=128, kernel=3, stride= 2, padding=1)
        self.layer2_1x1 = convrelu(in_channels=128, out_channels=128, kernel=1, stride=1, padding=0)

        # self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3 = convrelu(in_channels=128, out_channels=256, kernel=3, stride=2, padding=1)
        self.layer3_1x1 = convrelu(in_channels=256, out_channels=256, kernel=1, stride=1, padding=0)

        # self.layer4 = self.base_layers[7]  # sie=(N, 512, x.H/32, x.W/32)
        self.layer4 = convrelu(in_channels=256, out_channels=512, kernel=3, stride=2, padding=1)
        self.layer4_1x1 = convrelu(in_channels=512, out_channels=512, kernel=1, stride=1,  padding=0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(in_channels=256 + 512, out_channels=512, kernel=3, stride=1, padding=1)
        self.conv_up2 = convrelu(in_channels =128 + 512, out_channels=256, kernel=3, stride=1, padding=1)
        self.conv_up1 = convrelu(in_channels= 64 + 256, out_channels=256, kernel=3, stride=1, padding=1)
        self.conv_up0 = convrelu(in_channels= 64 + 256, out_channels=128, kernel=3, stride=1, padding=1)

        self.conv_original_size0 = convrelu(in_channels=in_channel, out_channels=64, kernel=3, stride=1, padding=1)
        self.conv_original_size1 = convrelu(in_channels=64, out_channels=64, kernel=3, stride=1, padding=1)
        self.conv_original_size2 = convrelu(in_channels=64 + 128, out_channels=in_channel, kernel=3, stride=1, padding=1)

        # self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat((x, layer3), dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat((x, layer2), dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat((x, layer1), dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat((x, layer0), dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat((x, x_original), dim=1)
        x = self.conv_original_size2(x)

        # out = self.conv_last(x)

        return  x

class ResnetUnet(nn.Module):
    def __init__(self, in_channel):
        super(ResnetUnet, self).__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)

        self.layer0_1x1 = convrelu(in_channels=64, out_channels=64, kernel=1, stride=1, padding=0)

        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(in_channels=64, out_channels=64, kernel=1, stride=1, padding=0)

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(in_channels=128, out_channels=128, kernel=1, stride=1, padding=0)

        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(in_channels=256, out_channels=256, kernel=1, stride=1, padding=0)

        self.layer4 = self.base_layers[7]  # sie=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(in_channels=512, out_channels=512, kernel=1, stride=1,  padding=0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(in_channels=256 + 512, out_channels=512, kernel=3, stride=1, padding=1)
        self.conv_up2 = convrelu(in_channels =128 + 512, out_channels=256, kernel=3, stride=1, padding=1)
        self.conv_up1 = convrelu(in_channels= 64 + 256, out_channels=256, kernel=3, stride=1, padding=1)
        self.conv_up0 = convrelu(in_channels= 64 + 256, out_channels=128, kernel=3, stride=1, padding=1)

        self.conv_original_size0 = convrelu(in_channels=in_channel, out_channels=64, kernel=3, stride=1, padding=1)
        self.conv_original_size1 = convrelu(in_channels=64, out_channels=64, kernel=3, stride=1, padding=1)
        self.conv_original_size2 = convrelu(in_channels=64 + 128, out_channels=in_channel, kernel=3, stride=1, padding=1)

        # self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat((x, layer3), dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat((x, layer2), dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat((x, layer1), dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat((x, layer0), dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat((x, x_original), dim=1)
        x = self.conv_original_size2(x)

        # out = self.conv_last(x)

        return  x



if __name__ == '__main__':
    gpu_id = 1


    model = viewTransformer(in_channel=15).cuda(gpu_id)
    inputImage = torch.randn((10, 15, 64, 64)).cuda(gpu_id)

    x = model(inputImage)

    print('check')