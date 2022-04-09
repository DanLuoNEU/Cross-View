import sys
from tqdm import tqdm
import scipy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from dataset.NTU_viewProjection import *
from dataset.NUCLAS_ViewProjection import *
from torch.utils.data import DataLoader
from torch.backends import cudnn

from utils import *
from modelZoo.actHeat import *


# import cv2
np.random.seed(0)
torch.manual_seed(0)
# torch.backends.cudnn.enabled = False

##### Configuration #####
# GPU
gpu_id = 2
num_workers = 2# 4
# Experiment
dataset = 'NUCLA'
modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'

# saveModel = os.path.join(modelRoot, dataset, 'action_viewTransformer_gray/')
#
# if not os.path.exists(saveModel):
#     os.makedirs(saveModel)

# DataLoader
if dataset == 'NUCLA':
    T = 0
    num_actions = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_res'
    trainSet = NUCLA_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='train', cam='1,2', T=T,
                                    target_view='view_1',
                                    project_view='view_2', test_view='view_3')
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    valSet = NUCLA_viewProjection(root_skeleton=root_skeleton,  root_list=path_list, phase='val',cam='1,2', T=T, target_view='view_1',
                              project_view='view_2', test_view='view_3')

    valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)

    # testSet = NUCLA_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='test', cam='1,2', T=T,
    #                                target_view='view_1', project_view='view_2', test_view='view_3')
    # #
    # testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=num_workers)


#
elif dataset == 'NTU-D':

    path_list = '/data/Yuexi/NTU-RGBD/list'
    root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    T = 20
    num_actions = 60
    trainSet = NTURGBD_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='train', T=T,
                                      target_view='C001', project_view='C002')
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    valSet = NTURGBD_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='val', T=T,
                                    target_view='C001', project_view='C002')

    valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)

'initialized action predictor'
# N = 2 * 40 ############ modify?!!
# P, Pall = gridRing(N)
# Drr = abs(P)
# DrrHeat = torch.from_numpy(Drr).float()
# Dtheta = np.angle(P)
# DthetaHeat = torch.from_numpy(Dtheta).float()

# net = heatAction(classNum=num_actions, Drr=DrrHeat, Dtheta=Dtheta, PRE=0, outRes=64, gpu_id=gpu_id)


'load pre-train model'
# modelPath = os.path.join(modelRoot, dataset, 'heatMapDYAN_view12')
modelPath = os.path.join(modelRoot, dataset, 'action_viewTransformer_gray')

stateDict = torch.load(os.path.join(modelPath, '150.pth'))['state_dict']
Drr = stateDict['sparseCodeGenerator.l1.rr']
Dtheta = stateDict['sparseCodeGenerator.l1.theta']


net = actionHeatTransformer(classNum=num_actions, Drr=Drr, Dtheta=Dtheta, PRE=0, outRes=64, gpu_id=gpu_id)

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# for param in net.sparseCodeGenerator.parameters():
#     param.requires_grad = False


net.load_state_dict(stateDict)

net.cuda(gpu_id)
net.eval()

################### Begin to Test #################
count = 0

with torch.no_grad():
    for i, sample in enumerate(valloader):


        'with view transformer'
        inputHeat = sample['heatmaps']
        inputImageSeq = sample['imageSequence']
        view = sample['info']['view']
        nframe = sample['info']['T_sample'].item()

        label = sample['action']

        # nframe = T  # for NTU-D

        'heatmap'
        # target_view = inputHeat[:, 0:nframe, :, :]
        # project_view = project_view_heat[:, 0:nframe, :, :]

        # 'gray image'
        target_view = inputImageSeq[:, 0:nframe, :, :, :]
        # project_view = project_view_image[:, 0:nframe, :, :, :]

        heatRes = inputHeat.shape[3]

        targetInput = target_view.cuda(gpu_id).reshape(target_view.shape[0], nframe, -1).float() #1xTx64x64
        projectInput = torch.zeros(targetInput.shape).cuda(gpu_id).float()

        pred = net(targetInput, projectInput, nframe, view, 'val')


        if label.item() == torch.argmax(pred).cpu().data.item():
            count = count + 1
        # totFrames = totFrames + nframe
        print('sample id:', i, 'label:', label.item(), 'out:', torch.argmax(pred).cpu().data.item())



    # print('total sample:', valSet.__len__(), 'correct pred:', count, 'acc:', count/valSet.__len__())



print('done')