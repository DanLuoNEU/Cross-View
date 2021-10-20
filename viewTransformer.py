import sys
from tqdm import tqdm
import scipy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.backends import cudnn
from torch.utils.data import DataLoader
from utils import *
from modelZoo.networks import *
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from dataset.NTU_viewProjection import *
from modelZoo.Unet import viewTransformer

# from NUCLAskeleton import *
def binarizedSparseCode(sparseCode):
    positiveC = (torch.pow(sparseCode, 2) + 1e-6) / (torch.sqrt(torch.pow(sparseCode, 2)) + 1e-6)
    sparseCode = torch.tanh(4*positiveC)

    return sparseCode


np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False

gpu_id = 3
num_workers = 2
PRE = 0
dataset = 'NUCLA'
#
modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'

saveModel = os.path.join(modelRoot, dataset, 'viewTransformer_heatmaps/')
if not os.path.exists(saveModel):
    os.makedirs(saveModel)

modelPath = os.path.join(modelRoot, dataset, 'heatMapDYAN_view12')
# modelPath = os.path.join(modelRoot, dataset, 'grayDYAN_view12')

stateDict = torch.load(os.path.join(modelPath, 'heatMapDYAN_view12100.pth'))['state_dict']
Drr = stateDict['l1.rr']
Dtheta = stateDict['l1.theta']
getSparseCode = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)
getSparseCode.load_state_dict(stateDict)
getSparseCode.cuda(gpu_id)

for param in getSparseCode.parameters():
    param.requires_grad = False

if dataset == 'NUCLA':
    T = 25
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_res'
    trainSet = NUCLA_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='train', cam='1,2', T=T, target_view='view_1',
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
    T = 25

    trainSet = NTURGBD_viewProjection(root_skeleton=root_skeleton, root_list=path_list,
                           phase='train', T=T, target_view='C001', project_view='C002')
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    valSet =  NTURGBD_viewProjection(root_skeleton=root_skeleton, root_list=path_list,
                 phase='val', T=T,target_view='C001', project_view='C002')

    valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)


# in_channel = Drr.shape[0]*4+1
in_channel = T
net = viewTransformer(in_channel=in_channel).cuda(gpu_id)
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


Epoch = 200

optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 250], gamma=0.1)
criterion = nn.MSELoss()

print('start training, project v2 to v1')

for epoch in range(1, Epoch+1):
    lossVal = []

    print('training epoch:', epoch)
    for i, sample in enumerate(trainloader):
        optimizer.zero_grad()

        target_view_heat = sample['target_view_heat']  # view_1
        project_view_heat = sample['project_view_heat'] # view_2

        """""
        target_view_image = sample['target_view_image']
        project_view_image = sample['project_view_image']

        targetFrame = sample['target_info']['T_sample']
        projectFrame = sample['project_info']['T_sample']

        if targetFrame <= projectFrame:
            nframe = targetFrame.item()
        else:
            nframe = projectFrame.item()  # for NUCLA

        # nframe = T  # for NTU-D

        'heatmap'
        # target_view = target_view_heat[:,0:nframe,:,:]
        # project_view = project_view_heat[:,0:nframe,:,:]
        target_view = target_view_heat[:, 0:nframe, :, :, :]
        project_view = project_view_heat[:, 0:nframe, :, :,:]

        'gray image'
        # target_view = target_view_image[:,0:nframe, :,:,:]
        # project_view = project_view_image[:,0:nframe, :, :, :]
        
        """
        target_view = target_view_heat.cuda(gpu_id).float().squeeze(0).permute(1, 0, 2, 3)
        project_view = project_view_heat.cuda(gpu_id).float().squeeze(0).permute(1, 0, 2, 3)
        heatRes = target_view.shape[3]
        njoint = target_view.shape[2]
        nframe = T


        """""
        targetInput = target_view.cuda(gpu_id).reshape(target_view.shape[0], nframe, -1).float()
        projectInput = project_view.cuda(gpu_id).reshape(project_view.shape[0], nframe, -1).float()

        target_MotionFeat, _ = getSparseCode.l1(targetInput, nframe)
        project_MotionFeat, _ = getSparseCode.l1(projectInput, nframe)

        target_MotionFeat_Bi = binarizedSparseCode(target_MotionFeat)  # Binarized motion feature
        project_MotionFeat_Bi = binarizedSparseCode(project_MotionFeat) # Binarized motion feature

        # project_MotionFeat_Bi = project_MotionFeat_Bi.reshape(project_MotionFeat_Bi.shape[0], project_MotionFeat_Bi.shape[1], heatRes, heatRes)  # 1 x T x 64 x 64
        # target_MotionFeat_Bi = target_MotionFeat_Bi.reshape(project_MotionFeat_Bi.shape)

        project_MotionFeat_Bi = project_MotionFeat_Bi.reshape(project_MotionFeat_Bi.shape[0], project_MotionFeat_Bi.shape[1], njoint, heatRes, heatRes)
        target_MotionFeat_Bi = target_MotionFeat_Bi.reshape(project_MotionFeat_Bi.shape)

        'permute dim: 20x312x64x64'
        project_MotionFeat_Bi = project_MotionFeat_Bi.squeeze(0).permute(1, 0, 2, 3)
        target_MotionFeat_Bi = target_MotionFeat_Bi.squeeze(0).permute(1, 0, 2, 3)
        
        out_MotionFeat_Bi = net(project_MotionFeat_Bi)  # 1xTx64x64

        loss = criterion(target_MotionFeat_Bi, out_MotionFeat_Bi)
        """
        out_heatmap = net(project_view)
        loss = criterion(out_heatmap, target_view)
        loss.backward()
        optimizer.step()
        # print('check')
        lossVal.append(loss.data.item())

    if epoch % 10 == 0:
        print('validating...')
        Error = []
        with torch.no_grad():
            for i, sample in enumerate(valloader):
                target_view_heat = sample['target_view_heat']  # view_1
                project_view_heat = sample['project_view_heat']  # view_2

                """""
                target_view_image = sample['target_view_image']
                project_view_image = sample['project_view_image']

                targetFrame = sample['target_info']['T_sample']
                projectFrame = sample['project_info']['T_sample']

                if targetFrame <= projectFrame:
                    nframe = targetFrame.item()
                else:
                    nframe = projectFrame.item()
                # nframe = T  # for NTU

                'heatmap'
                # target_view = target_view_heat
                # project_view = project_view_heat
                target_view = target_view_heat[:, 0:nframe, :, :, :]
                project_view = project_view_heat[:, 0:nframe, :, :, :]


                'gray image'
                # target_view = target_view_image[:,0:nframe, :,:,:]
                # project_view = project_view_image[:,0:nframe, :,:,:]

                heatRes = target_view.shape[3]
                njoint = target_view.shape[2]

                targetInput = target_view.cuda(gpu_id).reshape(target_view.shape[0], nframe, -1).float()
                projectInput = project_view.cuda(gpu_id).reshape(project_view.shape[0], nframe, -1).float()

                target_MotionFeat, _ = getSparseCode.l1(targetInput, nframe)
                project_MotionFeat, _ = getSparseCode.l1(projectInput, nframe)

                'binarized'
                target_MotionFeat_Bi = binarizedSparseCode(target_MotionFeat)  # Binarized motion feature
                project_MotionFeat_Bi = binarizedSparseCode(project_MotionFeat)  # Binarized motion feature

                # project_MotionFeat_Bi = project_MotionFeat_Bi.reshape(project_MotionFeat_Bi.shape[0],project_MotionFeat_Bi.shape[1], heatRes, heatRes)  # 1 x T x 64 x 64
                # target_MotionFeat_Bi = target_MotionFeat_Bi.reshape(project_MotionFeat_Bi.shape)

                project_MotionFeat_Bi = project_MotionFeat_Bi.reshape(project_MotionFeat_Bi.shape[0], project_MotionFeat_Bi.shape[1], njoint, heatRes, heatRes)
                target_MotionFeat_Bi = target_MotionFeat_Bi.reshape(project_MotionFeat_Bi.shape)

                'permute dim: 20x312x64x64'
                project_MotionFeat_Bi = project_MotionFeat_Bi.squeeze(0).permute(1, 0, 2, 3)
                target_MotionFeat_Bi = target_MotionFeat_Bi.squeeze(0).permute(1, 0, 2, 3)

                out_MotionFeat_Bi = net(project_MotionFeat_Bi)  # 1xTx64x64
                
                error = torch.norm(out_MotionFeat_Bi-target_MotionFeat_Bi)
                Error.append(error.data.item())
                """""
                target_view = target_view_heat.cuda(gpu_id).float().squeeze(0).permute(1, 0, 2, 3)
                project_view = project_view_heat.cuda(gpu_id).float().squeeze(0).permute(1, 0, 2, 3)
                out_heatmap = net(project_view)
                error = torch.norm(out_heatmap-target_view)
                Error.append(error.data.item())


        print('Epoch:', epoch, 'projection error:', np.mean(np.array(Error)))

    loss_val = np.mean(np.array(lossVal))
    print('Epoch:', epoch, '|loss:', loss_val)

    scheduler.step()

    if epoch % 20 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

print('done')
