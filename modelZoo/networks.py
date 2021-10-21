import torch
# torch.manual_seed(0)
import torch.nn as nn
import torchvision.models as models
# from modelZoo.convLSTM import*
from modelZoo.resNet import ResNet, Bottleneck, BasicBlock
from modelZoo.DyanOF import OFModel, creatRealDictionary
from modelZoo.sparseCoding import *
from modelZoo.actRGB import *
from st_gcn_net.st_gcn import Model
from modelZoo.actHeat import *
from utils import generateGridPoles, gridRing, fista, get_recover
import numpy as np


def load_preTrained_model(pretrained, newModel):
    'load pretrained resnet 50 to self defined model '
    'modified resnet has no last two layers, only return feature map'
    # resnet101 = models.resnet101(pretrained=True, progress=False)
    pre_dict = pretrained.state_dict()

    # modifiedResnet = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], zero_init_residual=False,
    #              groups=1, width_per_group=64, replace_stride_with_dilation=None,
    #              norm_layer=None)

    new_dict = newModel.state_dict()

    pre_dict = {k: v for k, v in pre_dict.items() if k in new_dict}

    new_dict.update(pre_dict)

    newModel.load_state_dict(new_dict)

    for param in newModel.parameters():
        param.requires_grad = False

    return newModel


class keyframeProposalNet(nn.Module):
    def __init__(self, numFrame, Drr, Dtheta, gpu_id, backbone):
        super(keyframeProposalNet, self).__init__()
        self.num_frame = numFrame
        self.gpu_id = gpu_id
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
                                    norm_layer=None) # Resent-18

        elif self.backbone == 'Resnet34':
            self.modifiedResnet = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], zero_init_residual=False,
                                         groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                         norm_layer=None)  # Resent-34

        """""
        self.Conv2d = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        """""
        self.relu = nn.LeakyReLU(inplace=True)

        # self.convLSTM = ConvLSTM(input_size=(7, 7), input_dim=512, hidden_dim=[256, 128, 64],
        #                          kernel_size=(3, 3), num_layers=3, gpu_id=self.gpu_id, batch_first=True, bias=True,
        #                          return_all_layers=False)
        'reduce feature map'
        self.layer2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        # self.layer2 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn_l4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # self.convBILSTM = ConvBILSTM(input_size=(7, 7), input_dim=512, hidden_dim=[256, 128, 64],
        #                          kernel_size=(3, 3), num_layers=3, gpu_id=self.gpu_id, batch_first=True, bias=True,
        #                          return_all_layers=False)
        self.Drr = nn.Parameter(Drr, requires_grad=True)
        self.Dtheta = nn.Parameter(Dtheta, requires_grad=True)

        # self.DYAN = OFModel(Drr, Dtheta, self.num_frame, self.gpu_id)
        'embeded info along time'
        # self.fc1 = nn.Linear(64 * self.num_frame, self.num_frame)
        # self.fc2 = nn.Linear(128*self.num_frame, self.num_frame)
        self.fcn1 = nn.Conv2d(self.num_frame, 25, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fcn2 = nn.Conv2d(25, 10, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # 64 x 10 x 3 x 3

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*10*3*3, self.num_frame)
        # self.fc = nn.Linear(128 * 10 * 3 * 3, self.num_frame)
        # self.drop = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()

    #     'add action recognition module'
    #
    # def conv_bn_layer(self, inputC, outputC, A, num_class):
    #     self.conv2d = nn.Conv2d(inputC, outputC, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1).cuda(self.gpu_id)
    #     self.bn = nn.BatchNorm2d(outputC, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True).cuda(self.gpu_id)
    #     self.fc_a = nn.Linear(outputC*A, num_class).cuda(self.gpu_id)
    #
    #     return self.conv2d, self.bn, self.fc_a

    def forward(self, x):
        imageFeature = self.modifiedResnet(x)
        Dictionary = creatRealDictionary(self.num_frame, self.Drr, self.Dtheta, self.gpu_id)
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
        feature = self.relu(x4)

        return feature, Dictionary, imageFeature

    def forward2(self, feature, alpha):
        x = feature.permute(1, 0, 2, 3)
        x = self.fcn1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fcn2(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.avg_pool(x)
        x = x.view(1, -1)
        x = self.fc(x)
        out = self.sig(alpha*x)
        return out

class onlineUpdate(nn.Module):
    def __init__(self, FRA, PRE, T, Drr, Dtheta, gpu_id):
        super(onlineUpdate, self).__init__()
        self.gpu_id = gpu_id
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.numFrame = T
        self.K_FPN = keyframeProposalNet(numFrame=self.numFrame, Drr=self.Drr, Dtheta=self.Dtheta, gpu_id=gpu_id, backbone='Resnet18')
        self.FRA = FRA
        self.PRE = PRE


        self.relu = nn.LeakyReLU(inplace=True)

        self.layer0 = nn.Conv2d(512*2, 512, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        # self.layer2 = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l0 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
        # self.layer2 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
        self.bn_l3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc = nn.Linear(1*64*3*3, 2)
    def get_keylist(self, x, alpha):
        feature, Dictionary, imgFeature = self.K_FPN.forward(x)
        indicator = self.K_FPN.forward2(feature, alpha)
        s = indicator[0, :]
        key_ind = (s > 0.995).nonzero().squeeze(1)
        key_list_tot = key_ind.cpu().numpy()

        # keys_to_pre = key_list[np.where(key_list >= self.FRA)[0]]
        # keylist_to_pred = list(keys_to_pre)  #key frames from K-FPN
        # key_list_FRA = list(set(key_list) - set(keylist_to_pred))

        key_list_FRA = list(key_list_tot[np.where(key_list_tot < self.FRA)[0]])  # input key list
        key_list = list(key_list_tot[np.where(key_list_tot < self.PRE+ self.FRA)[0]])
        keylist_to_pred = list(set(key_list) - set(key_list_FRA))


        Dict_key = Dictionary[key_list_FRA, :]
        feat_key = imgFeature[key_list_FRA, :]


        t, c, w, h = feat_key.shape
        feat_key = feat_key.reshape(1, t, c * w * h)
        sparseCode_key = fista(Dict_key, feat_key, 0.01, 100, self.gpu_id)

        return sparseCode_key, Dictionary, keylist_to_pred, key_list_FRA, key_list,imgFeature

        # return gtImgFeature, predImgFeature, keylist_to_pred


    def forward(self, imgFeature, sparseCode_key, Dictionary, fraNum):
        gtImgFeature = imgFeature[fraNum]
        c, w, h = gtImgFeature.shape
        newDictionary = torch.cat((Dictionary[0:self.FRA], Dictionary[fraNum].unsqueeze(0)))
        newImgFeature = torch.matmul(newDictionary, sparseCode_key).reshape(newDictionary.shape[0], c, w, h)

        predImgFeature = newImgFeature[-1]
        combineFeature = torch.cat((gtImgFeature, predImgFeature)).unsqueeze(0)
        x = self.layer0(combineFeature)
        x = self.bn_l0(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.bn_l1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.bn_l2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.bn_l3(x)
        x = self.relu(x)

        x = x.view(1, -1)
        out = self.fc(x)
        return out


class detectAction(nn.Module):
    def __init__(self, classNum, DrrImg, DthetaImg, DrrHeat, DthetaHeat, T, outRes, kinetics_pretrain, gpu_id):
        super(detectAction, self).__init__()
        self.gpu_id = gpu_id
        self.inLen = T
        self.classNum = classNum
        self.kinetics_pretrain = kinetics_pretrain
        # self.K_FPN = keyframeProposalNet(numFrame=self.inLen, Drr=DrrImg, Dtheta=DthetaImg, gpu_id=self.gpu_id, backbone='Resnet18')
        self.actionPredictorHeat = heatAction(classNum=self.classNum, Drr=DrrHeat, Dtheta=DthetaHeat, PRE=0, outRes=outRes, backbone='Resnet18', gpu_id=self.gpu_id)
        self.actionPredictorI3D = RGBAction(num_class=self.classNum, kinetics_pretrain=self.kinetics_pretrain)
        # self.actionPredictorGCN = Model(in_channels=3, num_class=self.classNum, graph_args={'layout': 'openpose', 'strategy':'spatial'}, edge_importance_weighting=True)
        # self.fuse = nn.Conv1d(2, 1, kernel_size=1, stride=1, padding=0, groups=1, bias=False, dilation=1)
    def forward(self, x, alpha, heatmaps, skeleton, nframe):

        # feature,_,_ = self.K_FPN.forward(x)
        # keyIndicator = self.K_FPN.forward2(feature, alpha)
        # keyInd = (keyIndicator[0,:] > 0.99).nonzero().squeeze(1)
        keyAction = False
        if_2S = True
        # # For all 3 streams
        # if keyAction:
        #     if nframe.item() < self.inLen:
        #         keyInd = keyInd[keyInd<=nframe.item()]
        #     else:
        #         keyInd = keyInd
        #     FRA = len(keyInd)
        # else:
        #     if nframe.item() < self.inLen:
        #         FRA = nframe.item()
        #     else:
        #         FRA = self.inLen
        #     keyInd = np.linspace(0, FRA-1, FRA, dtype=np.int)

        # For Only Heatmaps
        # FRA = self.inLen
        FRA = nframe
        keyInd = np.linspace(0, FRA-1, FRA, dtype=np.int)
        
        keyImage = x[keyInd].unsqueeze(0)
        keyHeatmaps = heatmaps[:, keyInd]
        # keySkeleton = skeleton[:, :, keyInd]    # N x C x T x V x numP, 1 x 2 x T x 15 x 1
        # keyImage = x[0:nframe].unsqueeze(0)
        # keyHeatmaps = heatmaps[:, 0:nframe]
        # FRA = nframe



        label_i3d = self.actionPredictorI3D(keyImage)

        # print('input heatsize:', keyHeatmaps.shape)

        label_heat = self.actionPredictorHeat(keyHeatmaps[:, 0:FRA], keyImage[:,0:FRA], FRA, 'heat')

        # label_gcn = self.actionPredictorGCN(keySkeleton)
        # label_i3d = None
        # label_heat = None

        if if_2S:
            # x_cat = torch.cat((label_heat, label_i3d)).unsqueeze(0)
            # label = self.fuse(x_cat).squeeze(0)
            label = 0.5 * label_i3d + 0.5 * label_heat
        # label = 0.2*label_i3d + 0.8*label_heat
        else:
            label = {'label_i3d': label_i3d, 'label_heat': label_heat}
            # label = {'label_i3d':label_i3d, 'label_heat':label_heat, 'label_gcn':label_gcn}
        # label = {'label_i3d': label_i3d}

        return label, FRA


class KFPNFullModel(nn.Module):
    def __init__(self,DrrImg, DthetaImg, DrrPose, DthetaPose, T, gpu_id, backbone):
        super(KFPNFullModel, self).__init__()
        self.gpu_id = gpu_id
        self.inLen = T
        self.backbone = backbone
        self.K_FPN = keyframeProposalNet(numFrame=self.inLen, Drr=DrrImg, Dtheta=DthetaImg, gpu_id=self.gpu_id, backbone=self.backbone)
        self.DrrPose = nn.Parameter(DrrPose, requires_grad=True)
        self.DthetaPose = nn.Parameter(DthetaPose, requires_grad=True)

    def forward(self, x, y, alpha, epoch):
        feature, Dictionary, _ = self.K_FPN.forward(x)
        out = self.K_FPN.forward2(feature, alpha*epoch)

        DictPose = creatRealDictionary(self.inLen, self.DrrPose, self.DthetaPose, self.gpu_id)

        # yhat = get_recover(DictPose, y, out)

        return feature, Dictionary, out, DictPose




if __name__ == "__main__":

    gpu_id = 2
    alpha = 4 # step size for sigmoid
    N = 4 * 40
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    # net = keyframeProposalNet(numFrame=40, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)

    # net = detectAction(numFrame=40, Drr=Drr, Dtheta=Dtheta, num_class=12, gpu_id=gpu_id)
    T = 40
    FRA = 30
    net = onlineUpdate(FRA=FRA, T=T, Drr=Drr, Dtheta=Dtheta, gpu_id=gpu_id)
    net.cuda(gpu_id)

    X = torch.randn(1, 40, 3, 224, 224).cuda(gpu_id)
    imF = []
    fcF = []
    temF =[]
    convF = []
    Y =[]
    for i in range(0, X.shape[0]):
        x = X[i]
        # feature, Dictionary, _ = net.forward(x)
        # out = net.forward2(feature, alpha)
        'keyframe prediction'
        for fraNum in range(FRA, T):

            gtImgFeature, predImgFeature, keylist_to_pred = net.get_keylist(x, alpha, fraNum)
            if fraNum in keylist_to_pre:
                label = 1
            else:
                label = 0

            out = net.forward(gtImgFeature, predImgFeature)
            print('check')



        'action recoginition'
        # pred = net.addActionModule(out, Dictionary, feature, num_class=20)

        # pred = net(x, alpha)



        print('check')

    print('done')
