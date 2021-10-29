import torch.nn as nn
import torch
import torch.nn.init as init
from modelZoo.sparseCoding import *
from utils import *
from modelZoo.actRGB import *
from modelZoo.gumbel_module import *
from scipy.spatial import distance
from modelZoo.binarize import Binarization

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


class binaryCoding(nn.Module):
    def __init__(self, num_binary):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(161, 64, kernel_size=(3,3), padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(64, 32, kernel_size=(3,3), padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(32, 64, kernel_size=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 500),
            # nn.Linear(64*26*8, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, num_binary)
        )

        for m in self.modules():
            # if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
            #     init.xavier_normal(m.weight.data)
            #     m.bias.data.fill_(0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class binarizeSparseCode(nn.Module):
    def __init__(self, num_binary, Drr, Dtheta, gpu_id, Inference, fistaLam):
        super(binarizeSparseCode, self).__init__()
        self.k = num_binary
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = Inference
        self.gpu_id = gpu_id
        self.fistaLam = fistaLam
        # self.sparsecoding = sparseCodingGenerator(self.Drr, self.Dtheta, self.PRE, self.gpu_id)
        # self.binaryCoding = binaryCoding(num_binary=self.k)
        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta, lam=fistaLam, gpu_id=self.gpu_id)
        self.binaryCoding = GumbelSigmoid()

    def forward(self, x, T):
        sparseCode, Dict = self.sparseCoding(x, T)
        # sparseCode = sparseCode.permute(2,1,0).unsqueeze(3)
        # # sparseCode = sparseCode.reshape(1, T, 20, 2)
        # binaryCode = self.binaryCoding(sparseCode)

        # reconstruction = torch.matmul(Dict, sparseCode)
        binaryCode = self.binaryCoding(sparseCode, force_hard=True, temperature=0.1, inference=self.Inference)

        # temp = sparseCode*binaryCode
        return binaryCode, sparseCode, Dict

class classificationHead(nn.Module):
    def __init__(self, num_class, Npole,dataType):
        super(classificationHead, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.dataType = dataType

        self.conv1 = nn.Conv2d(self.Npole, 128, (3,3), padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.gn1 = GroupNorm(128, 128)

        self.conv2 = nn.Conv2d(128, 64, (1,1))
        self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.gn2 = GroupNorm(64, 64)

        # self.conv3 = nn.Conv2d(64, 32, (1,1))
        # self.bn3 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.gn3 = GroupNorm(32, 32)
        self.conv3 = nn.Conv2d(64, 64, (1, 1))
        self.bn3 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.gn3 = GroupNorm(64, 64)

        self.conv4 = nn.Conv2d(64, 128, (1, 1))
        self.bn4 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.gn4 = GroupNorm(128, 128)

        self.RL = nn.LeakyReLU()
        # self.FC = nn.Linear(32*3*3, self.num_class)
        # self.FC = nn.Linear(32*20*4, self.num_class) # joint = 18
        # self.FC = nn.Linear(32*22*4, self.num_class) # joint = 20, 2D
        if self.dataType == '2D':
            self.FC = nn.Linear(128 * 27 * 4, self.num_class) # joint = 25, openpose
            # self.FC = nn.Linear(128 * 27 * 4, self.num_class)
            # self.FC = nn.Linear(128 * 17 *4, self.num_class)  # joing = 15, jhmdb
        else:
            # self.FC = nn.Linear(32 * 22 * 5, self.num_class) # joint=20, 3D
            # self.FC = nn.Linear(128 * 22 * 5, self.num_class)
            self.FC = nn.Linear(128 * 27 * 5, self.num_class)
        for m in self.modules():
            # if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
            #     init.xavier_normal(m.weight.data)
            #     m.bias.data.fill_(0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.gn1(x)
        x = self.RL(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.gn2(x)
        x = self.RL(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.gn3(x)
        x = self.RL(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.gn4(x)
        x = self.RL(x)

        # print('binary classifier fc size:', x.shape)
        x = x.view(x.size(0), -1)
        # x = self.FC(x)

        label = self.FC(x)
        return label


class classificationWBinarization(nn.Module):
    def __init__(self, num_class, Npole, num_binary, dataType):
        super(classificationWBinarization, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.num_binary = num_binary
        self.dataType = dataType
        self.BinaryCoding = binaryCoding(num_binary=self.num_binary)
        self.Classifier = classificationHead(num_class=self.num_class, Npole=Npole,dataType=self.dataType)

    def forward(self, x):
        'x is coefficients'
        inp = x.reshape(x.shape[0], x.shape[1], -1).permute(2,1,0).unsqueeze(-1)
        binaryCode = self.BinaryCoding(inp)
        binaryCode = binaryCode.t().reshape(self.num_binary, x.shape[-2], x.shape[-1]).unsqueeze(0)
        label = self.Classifier(binaryCode)

        return label,binaryCode

class classificationWSparseCode(nn.Module):
    def __init__(self, num_class, Npole, Drr, Dtheta, dataType,dim, gpu_id):
        super(classificationWSparseCode, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.Drr = Drr
        self.Dtheta = Dtheta
        # self.T = T
        self.gpu_id = gpu_id
        self.dim = dim
        self.dataType = dataType
        self.Classifier = classificationHead(num_class=self.num_class, Npole=Npole, dataType=self.dataType)
        # self.sparsecoding = sparseCodingGenerator(self.Drr, self.Dtheta, self.PRE, self.gpu_id) # old-version

        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta,lam=0.1, gpu_id=self.gpu_id)
    def forward(self, x, T):
        # sparseCode, Dict = self.sparsecoding.forward2(x, T) # old-version
        sparseCode, Dict = self.sparseCoding(x, T)
        Reconstruction = torch.matmul(Dict, sparseCode)   # sparseCode.detach()
        c = sparseCode.reshape(1, self.Npole, int(x.shape[-1]/self.dim), self.dim)
        label = self.Classifier(c)

        return label, Reconstruction


class Fullclassification(nn.Module):
    def __init__(self, num_class, Npole, num_binary, Drr, Dtheta,dim, dataType, Inference, gpu_id, fistaLam):
        super(Fullclassification, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.num_binary = num_binary
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = Inference
        self.gpu_id = gpu_id
        self.dim = dim
        self.dataType = dataType
        self.fistaLam = fistaLam
        # self.BinaryCoding = binaryCoding(num_binary=self.num_binary)
        self.BinaryCoding = GumbelSigmoid()
        #self.BinaryCoding = Binarization(self.Npole)
        self.Classifier = classificationHead(num_class=self.num_class, Npole=Npole, dataType=self.dataType)
        # self.sparsecoding = sparseCodingGenerator(self.Drr, self.Dtheta, self.PRE, self.gpu_id)
        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta,  lam=fistaLam, gpu_id=self.gpu_id)

    def get_sparse_stats(self, sparse_tensor, idx):
        
        sparse = sparse_tensor.clone()
        sparse = torch.abs(sparse)        
        print('sparse min', torch.min(sparse[0,:,idx]), flush=True)
        print('sparse max', torch.max(sparse[0,:,idx]), flush=True)
        print('sparse sort 1 sample', torch.sort(sparse[0,:,idx])[0], flush=True)

    def forward(self, x, T):
        # sparseCode, Dict = self.sparsecoding.forward2(x, T)
        sparseCode, Dict = self.sparseCoding(x, T)

        # inp = sparseCode.permute(2, 1, 0).unsqueeze(-1)
        # binaryCode = self.BinaryCoding(inp)

        # sparseCode = sparseCode**2
        # pdb.set_trace()
        sparseCode = sparseCode.detach()
        #binaryCode = self.BinaryCoding(sparseCode)
        binaryCode = self.BinaryCoding(sparseCode*2, force_hard=True, temperature=0.1, inference=self.Inference)

        temp1 = sparseCode * binaryCode
        # binaryCode = binaryCode.t().reshape(self.num_binary, int(x.shape[-1]/self.dim), self.dim).unsqueeze(0)
        # print('binarycode shape:', binaryCode.shape)
        temp = binaryCode.reshape(binaryCode.shape[0], self.Npole, int(x.shape[-1]/self.dim), self.dim)
        label = self.Classifier(temp)
        Reconstruction = torch.matmul(Dict, temp1)

        return label, binaryCode, Reconstruction, sparseCode

class twoStreamClassification(nn.Module):
    def __init__(self, num_class, Npole, num_binary, Drr, Dtheta, dim, gpu_id, dataType, kinetics_pretrain):
        super(twoStreamClassification, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.num_binary = num_binary
        self.Drr = Drr
        self.Dtheta = Dtheta
        # self.PRE = PRE
        self.gpu_id = gpu_id
        self.dataType = dataType
        self.dim = dim
        self.kinetics_pretrain = kinetics_pretrain

        self.dynamicsClassifier = Fullclassification(self.num_class, self.Npole, self.num_binary,
                                                          self.Drr, self.Dtheta, self.dim, self.dataType, self.gpu_id)
        self.RGBClassifier = RGBAction(self.num_class, self.kinetics_pretrain)

    def forward(self,skeleton, image, T, fusion):
        # stream = 'fusion'
        label1, binaryCode, Reconstruction = self.dynamicsClassifier(skeleton, T)
        label2 = self.RGBClassifier(image)

        if fusion:
            label = {'RGB':label1, 'Dynamcis':label2}
        else:
            label = 0.5 * label1 + 0.5 * label2

        # print('dyn:', label1, 'rgb:', label2)

        return label, binaryCode, Reconstruction

if __name__ == '__main__':
    gpu_id = 3
    # net = binaryCoding(num_binary=161).cuda(gpu_id)
    # net = Fullclassification()
    # x1 = torch.randn(20*3,161,1,1).cuda(gpu_id)
    # y1 = net(x1)
    # y1[y1>0] = 1
    # y1[y1<0] = -1
    #
    # x2 = torch.randn(1, 161, 1, 1).cuda(gpu_id)
    # y2 = net(x2)
    # y2[y2>0] = 1
    # y2[y2<0] = -1
    #
    # out_b1 = y1[0].detach().cpu().numpy().tolist()
    # out_b2 = y2[0].detach().cpu().numpy().tolist()
    # dist = distance.hamming(out_b1, out_b2)

    # net = classificationHead(num_class=10, Npole=161).cuda(gpu_id)
    # x = torch.randn(1, 161, 20, 3).cuda(gpu_id)
    #
    # y = net(x)
    N = 4*40
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    kinetics_pretrain = '../pretrained/i3d_kinetics.pth'
    net = twoStreamClassification(num_class=10, Npole=161, num_binary=161, Drr=Drr, Dtheta=Dtheta,
                                  PRE=0,dim=2, gpu_id=gpu_id, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
    x = torch.randn(1, 36, 50).cuda(gpu_id)
    xImg = torch.randn(1, 20, 3, 224, 224).cuda(gpu_id)
    T = x.shape[1]

    label, _, _ = net(x, xImg, T)




    print('check')






