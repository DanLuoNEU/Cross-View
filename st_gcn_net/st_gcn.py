import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from modelZoo.Convgru import ConvGRU
# from modelZoo.convLSTM import ConvLSTM
from st_gcn_net.utils.tgcn import ConvTemporalGraphical
from st_gcn_net.utils.graph import Graph
# from modelZoo.convLSTM import *
# from modelZoo.DyanOF import creatRealDictionary, OFModel
class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        'original'
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))
        """""
        'keep temporal length '
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 64, kernel_size, 1, **kwargs),
        ))
        """

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.CLS = nn.Conv2d(256, num_class, kernel_size=1)


    # def __init_weight(self):
    #     for m in self.modules():
    #         # if isinstance(m, nn.Conv1d):
    #         if isinstance(m, nn.Conv2d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, math.sqrt(2. / n))
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


    def forward(self, x):
        PRINT_GRADS = True
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)  # x.shape = N x 256 x t x 25

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])   # Nx256x1x1
        x = x.view(N, M, -1, 1, 1).mean(dim=1) # Nx256x1x1

        # prediction
        x = self.CLS(x) # N x 256 x num_class
        x = x.view(x.size(0), -1) # N x num_class
        # x.register_hook(
        #     lambda grad: print("fcn2.grad = {}".format(grad)) if PRINT_GRADS else False
        # )
        return x

    def extract_feature(self, x):
        PRINT_GRADS = True
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # feature.register_hook(
        #     lambda grad: print("feature.grad = {}".format(grad)) if PRINT_GRADS else False
        # )


        # prediction
        # x = self.fcn(x)
        # output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        # return _, feature
        return feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A




"""""
class train_feature_map(nn.Module):
    def __init__(self, in_channels, num_frame, numJoint, graph_args, edge_importance_weighting,
                 Drr, Dtheta, gpu_id):

        super(train_feature_map, self).__init__()
        self.gpu_id = gpu_id
        self.num_frame = num_frame # length of video
        self.numJoint = numJoint
        # self.if_bn = if_bn
        self.graph_setup = graph_args
        self.stgcn = Model(in_channels, self.num_frame, self.graph_setup, edge_importance_weighting)
        self.DYAN = OFModel(Drr, Dtheta, self.num_frame, self.gpu_id)
        # self.fc1 = nn.Linear(5040, 1024)
        # self.fc2 = nn.Linear(1024, self.num_frame)

    def forward(self, x):
        GraphicFeature = self.stgcn.extract_feature(x)  # input x: N x dim x T x num_joint x person
        # x = GraphicFeature.reshape(GraphicFeature.shape[0], self.num_frame, -1)
        x = GraphicFeature.squeeze(-1).permute(0,2,1,3)  # N x 60 x 64 x 13
        # x = nn.functional.avg_pool2d(x, (3, 3))
        # x = x.view(1,-1)
        # x = self.fc1(x)
        # x = self.fc2(x)  # N x T x 1

        return x

    def forward2(self, x):
        x = x.reshape(x.shape[0], self.num_frame, -1)  # reshape to N x T x (64*13)
        # x = x.unsqueeze(2)
        out = self.DYAN(x)

        return out

class exactact_keyframe(nn.Module):
    def __init__(self, in_channels, num_frame, numJoint, graph_args, edge_importance_weighting,
                 Drr, Dtheta, gpu_id):
        super(exactact_keyframe, self).__init__()
        self.gpu_id = gpu_id
        self.num_frame = num_frame  # length of video
        self.numJoint = numJoint
        # self.if_bn = if_bn
        self.graph_setup = graph_args
        self.stgcn = Model(in_channels, self.num_frame, self.graph_setup, edge_importance_weighting)
        self.Drr = nn.Parameter(Drr)
        self.Dtheta = nn.Parameter(Dtheta)
        self.DYAN = OFModel(Drr, Dtheta, self.num_frame, self.gpu_id)
        # self.fc1 = nn.Linear(5040, 1024)
        # self.fc2 = nn.Linear(1024, self.num_frame)
        self.fcn1 = nn.Conv2d(64, 32, kernel_size=1, stride=2)
        # self.fcn2 = nn.Conv2d(32, self.num_frame, kernel_size=1, stride=1)
        self.fcn2 = nn.Conv2d(32, 16, kernel_size=1, stride=2)
        # self.fc = nn.Linear(960, self.num_frame)   # for penn
        self.fc = nn.Linear(640, self.num_frame)
        self. bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.sig = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        GraphicFeature = self.stgcn.extract_feature(x)  # input x: N x dim x T x num_joint x person
        # x = GraphicFeature.reshape(GraphicFeature.shape[0], self.num_frame, -1)
        x = GraphicFeature.squeeze(-1).permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], self.num_frame, -1)  # reshape to N x T x (64*13)
        # x = x.unsqueeze(2)
        reconstFeature = self.DYAN(x)
        # _, Dictionary = self.DYAN.l1(x)
        Dictionary = creatRealDictionary(self.num_frame, self.Drr,  self.Dtheta, self.gpu_id)
        return Dictionary, GraphicFeature, reconstFeature

    def forward2(self, GraphicFeature, alpha):
        # GraphicFeature = self.stgcn.extract_feature(x)  # input x: N x dim x T x num_joint x person
        # x = GraphicFeature.reshape(GraphicFeature.shape[0], self.num_frame, -1)
        # x = GraphicFeature.squeeze(-1).permute(0, 2, 1, 3)
        # gcnFeature = nn.functional.avg_pool2d(x, (3, 3))

        # GraphicFeature: N x 64 x T x 13
        x = GraphicFeature.squeeze(-1)
        # x = nn.functional.avg_pool2d(x, x.size()[2:])  # NX64
        x = self.fcn1(x)
        x = self.bn1(x)
        x = self.fcn2(x)  # NX60x1x1
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.bn(x)
        out = x.view(x.size(0), -1)  # NX 60
        out = self.fc(out)
        out = self.sig(out * alpha)

        return out, x

    # def forward2(self, gcnFeature, alpha):
    #     x = gcnFeature.view(1, -1) # feature vector
    #     x = self.fc1(x)  #
    #     x = self.fc2(x)  # N x T x 1
    #     out = self.sig(x*alpha)

        # return out


class keyFrameBlock(nn.Module):
    def __init__(self, in_channels, num_frame, numJoint, graph_args, edge_importance_weighting, gpu_id):

        super(keyFrameBlock, self).__init__()
        self.gpu_id = gpu_id
        self.num_frame = num_frame # length of video
        self.numJoint = numJoint
        # self.if_bn = if_bn
        self.graph_setup = graph_args
        self.stgcn = Model(in_channels, self.num_frame, self.graph_setup, edge_importance_weighting)
        # self.convGRU = ConvGRU(input_size=num_frame, hidden_sizes=[32, 64, self.num_frame],
        #                             kernel_sizes=[3, 5, 3], n_layers=3, gpu_id=self.gpu_id)


        # self.convLSTM = ConvLSTM(input_size=(64, self.numJoint), input_dim=1, hidden_dim=[64, 128, 64],
        #                         kernel_size=(3,3), num_layers=3, gpu_id=gpu_id, batch_first=True, bias=True,
        #                          return_all_layers=False)


        self.fc1 = nn.Linear(self.numJoint*self.num_frame, 1024)
        self.fc2 = nn.Linear(1024, self.num_frame)
        # self.drop = nn.Dropout(0.5)
        self.sig = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d(1)
        # if if_define_init:
        # self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv1d):
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                # m.weight.data.fill_(0.01)
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)
    def forward(self, x, alpha):
        PRINT_GRADS = True
        GraphicFeature = self.stgcn.extract_feature(x)  # graph feature

        if GraphicFeature.shape[-1] > 1: # for multi-person senario
            out = []
            for i in range(0, GraphicFeature.shape[-1]):
                gcnFeature = GraphicFeature[:,:,:,:, i].permute(0, 2, 1, 3).unsqueeze(2) # N x T x 1 x 64 x 17
                'use covelstm'
                temporalFeature, _ = self.convLSTM(gcnFeature)
                x = temporalFeature[0].permute(0, 2, 3, 1, 4)
                x = x.contiguouse().view(x.shape[0], x.shape[1], x.shape[2], -1).permute(0, 3, 1, 2) # N x (50*17) x 32 x 64
                x = nn.functional.avg_pool2d(x, x.size()[2:])  # N x 850x1x1
                x = x.view(1, -1)
                out = self.fc1(x)
                # out = self.drop(out)
                out = self.fc2(out)

                'convGRU'
                # temporalFeature = self.convGRU(gcnFeature)
                # temFeature.append(temporalFeature)
                # x = temporalFeature.view(temporalFeature.size(0), -1)

                out = self.sig(alpha * out)

                out.append(out)

        else:
            gcnFeature = GraphicFeature.squeeze(-1).permute(0, 2, 1, 3).unsqueeze(2)  # N x T x 256 x 17
            # temporalFeature = self.convGRU(gcnFeature)
            # x = temporalFeature.view(temporalFeature.size(0), -1) # N x T
            temporalFeature, _ = self.convLSTM(gcnFeature)
            x = temporalFeature[0].permute(0, 2, 3, 1, 4)
            x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2], -1).permute(0, 3, 1,
                                                                                     2)  # N x (50*17) x 32 x 64
            x = nn.functional.avg_pool2d(x, x.size()[2:])  # N x 850x1x1
            x = x.view(1, -1)
            out = self.fc1(x)
            # out = self.drop(out)
            out = self.fc2(out)

            'add bn'
            # if self.if_bn:
            #     out = out.view(out.size(0), 1, -1)
            #     out = self.bn(out)
            #     out = out.view(out.size(0), -1)


            out = self.sig( alpha * out)

            # out.register_hook(
            #     lambda grad: print("keyframeOut.grad = {}".format(grad)) if PRINT_GRADS else False
            # )

        # x = x.view(x.size(0), 1, -1)
        # x = self.bn(x)
        # x = x.view(x.size(0), -1)
        # out.register_hook(
        #     lambda grad: print("keyframeOut.grad = {}".format(grad)) if PRINT_GRADS else False
        # )
        return out
"""""

if __name__ == '__main__':
    gpu_id = 1
    # model = Model(3, 60, {'layout': 'ntu-rgb+d', 'strategy':'spatial'}, edge_importance_weighting=True)
    # modelPath = './checkpoints/st_gcn.ntu-xsub-300b57d4.pth'
    # stateDict = torch.load(modelPath)
    # model.load_state_dict(stateDict, strict=False)
    num_frame = 50
    numJoint = 17
    keyFrameModel = keyFrameBlock(3, num_frame, numJoint, {'layout': 'hm3.6', 'strategy':'distance'}, edge_importance_weighting=True,
                                  gpu_id=gpu_id)
    # net = train_feature_map(2, num_frame, numJoint,{'layout': 'hm3.6', 'strategy':'distance'},edge_importance_weighting=True,Drr=1,Dtheta=1,gpu_id=gpu_id  )
    x = torch.randn(1, 3, num_frame, 17, 1).cuda(gpu_id)
    keyFrameModel.cuda(gpu_id)
    out = keyFrameModel(x, 1)
    print(out)
    # out = model.forward(x)
    # _, feature = model.extract_feature(x)
    # print(feature.shape)
