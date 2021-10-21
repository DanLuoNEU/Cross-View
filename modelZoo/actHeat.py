import torch
import torch.nn as nn
import torch.nn.functional as F
from modelZoo.sparseCoding import sparseCodingGenerator
from modelZoo.resNet import ResNet, Bottleneck, BasicBlock
from modelZoo.Unet import viewTransformer
class ActionClass(nn.Module):
    def __init__(self, classNum, inputCh):
        super(ActionClass, self).__init__()

        self.classNum = classNum

        self.conv_b1_1 = nn.Conv2d(inputCh, 128, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.batchnorm_b1_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.groupNorm_b1_1 = GroupNorm(128, 128)
        self.conv_b1_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.batchnorm_b1_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.groupNorm_b1_2 = GroupNorm(256, 256)

        self.conv_b2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.batchnorm_b2_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.groupNorm_b2_1 = GroupNorm(256, 256)
        self.conv_b2_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.batchnorm_b2_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.groupNorm_b2_2 = GroupNorm(512, 512)

        self.conv_b3_1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=1, bias=False, dilation=1)
        self.batchnorm_b3_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.groupNorm_b3_1 = GroupNorm(512, 512)
        self.conv_b3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.batchnorm_b3_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.groupNorm_b3_2 = GroupNorm(512, 512)

        self.relu = nn.LeakyReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(8)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, self.classNum)

    def forward(self, x):
        x = self.conv_b1_1(x)
        x = self.batchnorm_b1_1(x)
        # x = self.groupNorm_b1_1(x)

        x = self.relu(x)


        x = self.conv_b1_2(x)
        x = self.batchnorm_b1_2(x)
        # x = self.groupNorm_b1_2(x)
        x = self.relu(x)


        x = self.conv_b2_1(x)
        x = self.batchnorm_b2_1(x)
        # x = self.groupNorm_b2_1(x)
        x = self.relu(x)

        x = self.conv_b2_2(x)
        x = self.batchnorm_b2_2(x)
        # x = self.groupNorm_b2_2(x)
        x = self.relu(x)


        x = self.conv_b3_1(x)
        x = self.batchnorm_b3_1(x)
        # x = self.groupNorm_b3_1(x)
        x = self.relu(x)



        x = self.conv_b3_2(x)
        x = self.batchnorm_b3_2(x)
        # x = self.groupNorm_b3_2(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(1,-1)
        x = self.fc(x)
        # x = F.softmax(x,dim=0)

        return x

class GroupNorm(nn.Module):
    r"""Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization`_ .
    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    This layer uses statistics computed from input data in both training and
    evaluation modes.
    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)
    Examples::
        # >>> input = torch.randn(20, 6, 10, 10)
        # >>> # Separate 6 channels into 3 groups
        # >>> m = nn.GroupNorm(3, 6)
        # >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        # >>> m = nn.GroupNorm(6, 6)
        # >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        # >>> m = nn.GroupNorm(1, 6)
        # >>> # Activating the module
        # >>> output = m(input)
    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    # __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)



class heatAction(nn.Module):
    def __init__(self, classNum, Drr, Dtheta,PRE, outRes ,backbone, gpu_id):
        super(heatAction, self).__init__()
        self.sparseCodeGenerator = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)
        self.inputCh = Drr.shape[0]*4 + 1
        self.labelPredictor = ActionClass(classNum, self.inputCh)
        self.outRes = outRes
        self.backone = backbone
        self.imageFeatExtractor = imageFeatureExtractor(self.backone)

    # def forward(self, imageSeq):
    #     feature, _ = self.imageFeatExtractor(imageSeq)

    def forward(self, objHeat, imageSeq, T, useStream):


        if useStream == 'rgb':
            # feature, _ = self.imageFeatExtractor(imageSeq)
            # print('feature size:', feature.shape)
            objHeat = imageSeq.reshape(T, -1).unsqueeze(0)   # T x 64 x 64
        # elif useStream == '2S':
        #     feature, _ = self.imageFeatExtractor(imageSeq)
        #     objHeat_rgb = feature.reshape(T, -1).unsqueeze(0)
        #     objHeat = torch.cat

        else:
            objHeat = objHeat

        # print(type(objHeat))

        sparseCode_obj = self.sparseCodeGenerator.forward2(objHeat, T)
        sparseCode_obj = sparseCode_obj.reshape(sparseCode_obj.shape[0],sparseCode_obj.shape[1], self.outRes, self.outRes)
        # if self.if_context == True:
        #     sparseCode_context = self.sparseCodeGenerator.forward2(contextHeat, T)
        #     sparseCode_context = sparseCode_context.reshape(sparseCode_context.shape[0],sparseCode_context.shape[1], self.outRes, self.outRes)
        #
        #     sparseCode = torch.cat((sparseCode_obj, sparseCode_context), 1)
        #
        # else:
        sparseCode = sparseCode_obj


        'binarize sparse code'
       
        positiveC = (torch.pow(sparseCode,2)+1e-6)/(torch.sqrt(torch.pow(sparseCode, 2))+1e-6)
        sparseCode = torch.tanh(4*positiveC)

        label = self.labelPredictor.forward(sparseCode)

        return label


class imageFeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(imageFeatureExtractor, self).__init__()
        self.backbone = backbone
        if self.backbone == 'Resnet50':
            self.modifiedResnet = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], zero_init_residual=False,
                                         groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                         norm_layer=None)  # Resnet-50

            self.Conv2d = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
            self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        elif self.backbone == 'Resnet18':
            self.modifiedResnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False,
                                         groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                         norm_layer=None)  # Resent-18

        elif self.backbone == 'Resnet34':
            self.modifiedResnet = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], zero_init_residual=False,
                                         groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                         norm_layer=None)  # Resent-34

        self.relu = nn.LeakyReLU(inplace=True)

        'reduce feature map'
        self.layer2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        # self.layer2 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn_l3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4 = nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn_l4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        imageFeature = self.modifiedResnet(x)
        if self.backbone == 'Resnet50':
            convx = self.Conv2d(imageFeature)
            convx = self.bn1(convx)
            convx = self.relu(convx)
        else:

            convx = imageFeature

        x2 = self.layer2(convx)
        x2 = self.bn_l2(x2)
        x2 = self.relu(x2)

        x3 = self.layer3(x2)
        x3 = self.bn_l3(x3)
        x3 = self.relu(x3)
        # feature = x3

        x4 = self.layer4(x3)
        x4 = self.bn_l4(x4)
        reducedFeature = self.relu(x4)

        return reducedFeature, imageFeature


class actionHeatTransformer(nn.Module):
    def __init__(self, classNum, Drr, Dtheta, PRE, outRes,gpu_id):
        super(actionHeatTransformer, self).__init__()
        self.sparseCodeGenerator = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)
        self.inputChanel = Drr.shape[0] * 4 + 1

        self.viewTransformer = viewTransformer(in_channel=self.inputChanel)
        self.labelPredictor = ActionClass(classNum, self.inputChanel)

        self.outRes = outRes

        # self.pool = nn.functional.adaptive_max_pool2d

    def get_binaryC(self, x):
        positiveC = (torch.pow(x, 2) + 1e-6) / (torch.sqrt(torch.pow(x, 2)) + 1e-6)
        sparseCode = torch.tanh(4 * positiveC)
        return sparseCode

    def forward(self, x_view1, x_view2, T, view, phase):

        if phase == 'train':
            sparseCode_view1,_ = self.sparseCodeGenerator.forward2(x_view1, T)

            sparseCode_view1 = sparseCode_view1.reshape(sparseCode_view1.shape[0], sparseCode_view1.shape[1], self.outRes, self.outRes)

            sparseCode_view2,_ = self.sparseCodeGenerator.forward2(x_view2, T)

            sparseCode_view2 = sparseCode_view2.reshape(sparseCode_view1.shape)

            'binarize sparse code'

            binarySparseCode_view1 = self.get_binaryC(sparseCode_view1)
            binarySparseCode_view2 = self.get_binaryC(sparseCode_view2)

            projected_view21 = self.viewTransformer(binarySparseCode_view2)   # from view2 to view1

            # temp = torch.cat((binarySparseCode_view1, projected_view21),0)

            label1 = self.labelPredictor.forward(binarySparseCode_view1)
            label2 = self.labelPredictor.forward(projected_view21)

            label = 0.5*label1 + 0.5*label2

        elif phase == 'val':

            sparseCode, _ = self.sparseCodeGenerator.forward2(x_view1, T)
            sparseCode = sparseCode.reshape(sparseCode.shape[0], sparseCode.shape[1], self.outRes, self.outRes)
            binarySparseCode = self.get_binaryC(sparseCode)
            # label = self.labelPredictor(binarySparseCode)
            if view == 'view_1':
                label = self.labelPredictor(binarySparseCode)
            else:
                binarySparseCode_trans = self.viewTransformer(binarySparseCode)
                label = self.labelPredictor(binarySparseCode_trans)
        else:
            sparseCode_view2, _ = self.sparseCodeGenerator.forward2(x_view2, T)
            sparseCode_view2 = sparseCode_view2.reshape(sparseCode_view2.shape[0], sparseCode_view2.shape[1], self.outRes, self.outRes)

            binarySparseCode_view2 = self.get_binaryC(sparseCode_view2)

            projected_view = self.viewTransformer(binarySparseCode_view2)

            label = self.labelPredictor.forward(projected_view)
        return label



# if __name__ == '__main__':



