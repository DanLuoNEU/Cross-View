import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

from modelZoo.i3dpt import I3D, I3D_head


class BaseNet(nn.Module):
    """
    Backbone network of the model
    """

    def __init__(self, base_name, data_type, kinetics_pretrain):

        super(BaseNet, self).__init__()

        self.base_name = base_name
        # self.kinetics_pretrain = cfg.kinetics_pretrain
        self.kinetics_pretrain = kinetics_pretrain
        self.freeze_stats = True
        self.freeze_affine = True
        self.fp16 = False
        self.data_type = data_type

        if self.base_name == "i3d":
            self.base_model = build_base_i3d(self.data_type, self.kinetics_pretrain, self.freeze_affine)
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Applies network layers on input images

        Args:
            x: input image sequences. Shape: [batch_size, T, C, W, H]
        """

        x = x.permute(0, 2, 1, 3, 4)  # [N,T,C,W,H] --> [N,C,T,W,H]
        conv_feat = self.base_model(x)

        # reshape to original size
        conv_feat = conv_feat.permute(0, 2, 1, 3, 4)  # [N,C,T,W,H] --> [N,T,C,W,H]

        return conv_feat


def build_base_i3d(data_type, kinetics_pretrain=None, freeze_affine=True):
    # print("Building I3D model...")

    i3d = I3D(num_classes=400, data_type=data_type)
    # kinetics_pretrain = '/pretrained/i3d_flow_kinetics.pth'
    if kinetics_pretrain is not None:
        if os.path.isfile(kinetics_pretrain):
            # print("Loading I3D pretrained on Kinetics dataset from {}...".format(kinetics_pretrain))
            print('Loading pretrained I3D:')
            i3d.load_state_dict(torch.load(kinetics_pretrain))
        else:
            raise ValueError("Kinetics_pretrain doesn't exist: {}".format(kinetics_pretrain))

    base_model = nn.Sequential(i3d.conv3d_1a_7x7,
                               i3d.maxPool3d_2a_3x3,
                               i3d.conv3d_2b_1x1,
                               i3d.conv3d_2c_3x3,
                               i3d.maxPool3d_3a_3x3,
                               i3d.mixed_3b,
                               i3d.mixed_3c,
                               i3d.maxPool3d_4a_3x3,
                               i3d.mixed_4b,
                               i3d.mixed_4c,
                               i3d.mixed_4d,
                               i3d.mixed_4e,
                               i3d.mixed_4f)

    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad = False

    if freeze_affine:
        base_model.apply(set_bn_fix)


    for p in base_model.parameters():
        p.requires_grad = False


    return base_model

def build_conv(base_name='i3d', kinetics_pretrain=None, mode='global', freeze_affine=True):

    if base_name == "i3d":

        i3d = I3D_head()

        model_dict = i3d.state_dict()
        if kinetics_pretrain is not None:
            if os.path.isfile(kinetics_pretrain):
                print ("Loading I3D head pretrained")
                pretrained_dict = torch.load(kinetics_pretrain)
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                i3d.load_state_dict(model_dict)
            else:
                raise ValueError ("Kinetics_pretrain doesn't exist: {}".format(kinetics_pretrain))
        #
        # if mode == 'context':
        #     # for context net
        model = nn.Sequential(i3d.maxPool3d,
                                  i3d.mixed_5b,
                                  i3d.mixed_5c,
                              i3d.avg_pool)
        # else:
        #     # for global branch
        # model = nn.Sequential(i3d.mixed_5b,
        #                       i3d.mixed_5c)

    else:
        raise NotImplementedError

    def set_bn_fix(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

    if freeze_affine:
        model.apply(set_bn_fix)

    return model



class RGBAction(nn.Module):
    def __init__(self, num_class, kinetics_pretrain):

        super(RGBAction, self).__init__()
        self.num_class = num_class
        self.base_net = 'i3d'
        self.data_type = 'rgb'
        self.freeze_stats = True
        self.freeze_affine = True
        self.fc_dim = 256
        self.dropout_prob = 0.3
        self.pool_size = 14
        self.fp16 = False
        self.kinetics_pretrain = kinetics_pretrain

        self.featureExtractor = BaseNet(self.base_net, self.data_type, self.kinetics_pretrain)
        self.i3d_conv = build_conv(self.base_net, self.kinetics_pretrain, 'global', self.freeze_affine)
        # for param in self.featureExtractor.parameters():
        #     param.requires_grad = False

        # for param in self.i3d_conv.parameters():
        #     param.requires_grad = False

        self.layer1 = nn.Conv3d(1024, self.fc_dim,
                                    kernel_size=1, stride=1, bias=True)

        # self.global_cls = nn.Conv3d(
        #         self.fc_dim * self.pool_size**2,
        #         self.num_class,
        #         (1,1,1),
        #         bias=True)

        self.global_cls = nn.Conv3d(self.fc_dim,self.num_class,(1,1,1), bias=True )




        self.dropout = nn.Dropout(self.dropout_prob)


    def forward(self, x):
        'global_feat: 1xTx512x7x7'

        STfeature = self.featureExtractor(x)

        N, T, _,_,_ = STfeature.size()

        STconvFeat = self.i3d_conv(STfeature.permute(0, 2, 1, 3, 4))
        STconvFeat = self.layer1(STconvFeat)
        # STconvFeat_flat = STconvFeat.permute(0, 2, 1, 3, 4).contiguous().view(N, T, -1, 1, 1)
        # STconvFeat_flat = STconvFeat_flat.permute(0, 2, 1, 3, 4).contiguous()


        STconvFeat_final = self.dropout(STconvFeat)


        global_class = self.global_cls(STconvFeat_final)
        global_class = global_class.squeeze(3)
        global_class = global_class.squeeze(3)
        global_class = global_class.mean(2)

        return global_class



if __name__ == '__main__':
    gpu_id = 1
    kinetics_pretrain = '../pretrained/i3d_kinetics.pth'
    net = RGBAction(num_class=12, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
    inputImage = torch.randn(1, 40, 3, 224, 224).cuda(gpu_id)
    # inputData = torch.randn(1, 40, 512, 7, 7).cuda(gpu_id)
    # inputheatData = torch.randn(1, 40, 64, 64).cuda(gpu_id)

    # baseNet = BaseNet('i3d', 'rgb').cuda(gpu_id)
    # i3dFeature = baseNet(inputImage)
    pred = net(inputImage)

    print(pred.shape)