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

saveModel = os.path.join(modelRoot, dataset, 'action_viewTransformer_gray/')

if not os.path.exists(saveModel):
    os.makedirs(saveModel)

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
    #
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
modelPath = os.path.join(modelRoot, dataset, 'grayDYAN_view12')

stateDict = torch.load(os.path.join(modelPath, '100.pth'))['state_dict']
Drr = stateDict['l1.rr']
Dtheta = stateDict['l1.theta']


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

transModelPath = os.path.join(modelRoot, dataset, 'viewTransformer_grayImg')
transDict = torch.load(os.path.join(transModelPath, 'viewTransformer_grayImg200.pth'))['state_dict']
newDict = net.viewTransformer.state_dict()
dicts = {k: v for k, v in transDict.items() if k in newDict}
newDict.update(dicts)
net.viewTransformer.load_state_dict(newDict)
# for param in net.viewTransformer.parameters():
#     param.requires_grad = False

net.cuda(gpu_id)

################### Optimization and Criterion#################
alpha = 3

Epoch = 150

# lr_heat, lr_i3d, lr_gcn = 1e-3, 5e-5, 1e-3
# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.0001, momentum=0.9)
# optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.actionPredictorHeat.parameters()), 'lr':lr_heat},
#                              {'params':filter(lambda x: x.requires_grad, net.actionPredictorI3D.parameters()), 'lr':lr_i3d},
#                              {'params':filter(lambda x: x.requires_grad, net.actionPredictorGCN.parameters()), 'lr':lr_gcn}],
#                             weight_decay=0.0001, momentum=0.9)


optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
criterion = nn.CrossEntropyLoss()
######################## Main #################################
print(f'start training... fine-tune pretrain model')
for epoch in range(0, Epoch+1):
    lossVal = []
    loss_I3D = []
    loss_Heat = []
    loss_fusion = []
    print('training epoch:', epoch)
    # net.train()
    # pbar = tqdm(total=len(trainloader),file=sys.stdout)
    # num_display = int(len(trainloader)/5)
    for i, sample in enumerate(trainloader):
        optimizer.zero_grad()
        ''
        # img_data = sample['imgSequence_to_use'][0]
        # skeleton = sample['sequence_to_use']
        # heatmaps = sample['heatmap_to_use']
        # label = sample['actLabel'].cuda(gpu_id)
        # if T==0:
        #     nframe = sample['info']['T_sample'][0]
        #     # T_heat =
        # else:
        #     nframe = sample['nframes']
        #
        # inputImgData = img_data[0].cuda(gpu_id).float()
        # inputHeatmaps = heatmaps.cuda(gpu_id).reshape(heatmaps.shape[0], nframe, -1).float()
        # inputPoseData = skeleton.permute(0,3,1,2).unsqueeze(4).cuda(gpu_id).float()

        # print('sample id:', i, 'img shape:', inputImgData.shape, 'pose shape:', inputPoseData.shape)
        # pred,_ = net(inputImgData, alpha, inputHeatmaps, inputPoseData, nframe)
        # pred = net(inputHeatmaps, nframe.item())

        'with view transformer'
        target_view_heat = sample['target_view_heat']  # view_1
        project_view_heat = sample['project_view_heat']  # view_2

        target_view_image = sample['target_view_image']
        project_view_image = sample['project_view_image']

        targetFrame = sample['target_info']['T_sample']
        projectFrame = sample['project_info']['T_sample']

        label = sample['action'].cuda(gpu_id)

        if targetFrame <= projectFrame:
            nframe = targetFrame.item()
        else:
            nframe = projectFrame.item()  # for NUCLA

        # nframe = T  # for NTU-D

        'heatmap'
        # target_view = target_view_heat[:, 0:nframe, :, :]
        # project_view = project_view_heat[:, 0:nframe, :, :]

        # 'gray image'
        target_view = target_view_image[:, 0:nframe, :, :, :]
        project_view = project_view_image[:, 0:nframe, :, :, :]

        heatRes = target_view.shape[3]

        targetInput = target_view.cuda(gpu_id).reshape(target_view.shape[0], nframe, -1).float()  #1xTx64x64
        projectInput = project_view.cuda(gpu_id).reshape(project_view.shape[0], nframe, -1).float()



        pred = net(targetInput, projectInput, nframe,[],'train')

        loss = criterion(pred, label)
        # loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        # print('check')
        lossVal.append(loss.data.item())
        # loss_I3D.append(loss_i3d.item())
        # loss_Heat.append(loss_heat.item())
        # loss_fusion.append(criterion(pred['label_heat'], out_i3d).item())

    if epoch % 5 == 0:
        print('validating...')
        count = 0
        keyFrames = []
        # totFrames = 0
        with torch.no_grad():
            # net_val = net.eval()

            for i, sample in enumerate(valloader):

                """""
                img_data = sample['imgSequence_to_use']
                skeleton = sample['sequence_to_use']
                # skeleton = sample['baseline_to_use']

                heatmaps = sample['heatmap_to_use']
                label = sample['actLabel'].cuda(gpu_id)
                if T==0:
                    nframe = sample['info']['T_sample']
                else:
                    nframe = sample['nframes']

                inputImgData = img_data[0].cuda(gpu_id).float()
                inputHeatmaps = heatmaps.cuda(gpu_id).reshape(heatmaps.shape[0], nframe, -1).float()
                inputPoseData = skeleton.permute(0, 3, 1, 2).unsqueeze(4).cuda(gpu_id).float()

                # pred_dict, numKey = net(inputImgData, alpha, inputHeatmaps, inputPoseData, nframe)
                pred = net(inputHeatmaps, nframe.item())
                """

                """""
                'with view transformer'

                test_view_heat = sample['test_view_heat']  # view_1

                test_view_image = sample['test_view_image']

                testFrame = sample['test_info']['T_sample']
                
                nframe = testFrame.item()
                """

                inputHeat = sample['heatmaps']
                inputImageSeq = sample['imageSequence']
                view = sample['info']['view']
                nframe = sample['info']['T_sample'].item()

                label = sample['action']


                  # for NUCLA

                # nframe = T  # for NTU-D

                'heatmap'
                # test_view = test_view_heat[:, 0:nframe, :, :]

                # 'gray image'
                # test_view = test_view_image[:, 0:nframe, :, :, :]
                # target_view = target_view_image[:, 0:nframe, :, :, :]
                # project_view = project_view_image[:, 0:nframe, :, :, :]
                target_view = inputImageSeq[:, 0:nframe, :, :, :]

                heatRes = target_view.shape[3]

                testInput = target_view.cuda(gpu_id).reshape(target_view.shape[0], nframe, -1).float()  # 1xTx64x64
                targetInput = torch.zeros(testInput.shape).cuda(gpu_id)

                pred = net(testInput, targetInput, nframe,[], 'val')

                if label.item() == torch.argmax(pred).cpu().data.item():
                    count = count + 1
                # totFrames = totFrames + nframe
                # print('sample id:', i, 'label:', label.item(), 'out:',
                #         torch.argmax(pred).cpu().data.item())



            print('total sample:', valSet.__len__(), 'correct pred:', count, 'acc:', count/valSet.__len__())

    loss_val = np.mean(np.array(lossVal))
    print('Epoch:', epoch, '|loss:', loss_val)
    # print('Epoch:', epoch, '|loss:', loss_val, '|loss_i3d:', np.mean(np.array(loss_I3D)), '|loss_heat:', np.mean(np.array(loss_Heat)),
    #       '|loss_fusion:', np.mean(np.array(loss_fusion)))

    scheduler.step()
    # if epoch % 10 == 0:
    #     torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
    #                 'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

print('done')