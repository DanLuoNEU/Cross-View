import sys
from tqdm import tqdm
import scipy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.backends import cudnn

from utils import *
from modelZoo.networks import *

from dataset.NUCLAsubject import *
from dataset.NUCLAS_ViewProjection import *
from dataset.NTU_viewProjection import *

# import cv2
np.random.seed(0)
torch.manual_seed(0)
# torch.backends.cudnn.enabled = False

##### Configuration #####
# GPU
gpu_id = 1
num_workers = 4# 4
# Experiment
  #8 #40
dataset = 'N-UCLA'
modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'

saveModel = os.path.join(modelRoot, dataset, 'viewTransformer_grayImg')
if not os.path.exists(saveModel):
    os.makedirs(saveModel)

    os.makedirs(saveModel)
# DataLoader
if dataset == 'NUCLA':
    T = 0
    num_actions = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_res'
    trainSet = NUCLA_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='train', cam='1,2', T=T, target_view='view_1',
                              project_view='view_2', test_view='view_3')
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    # valSet = NUCLA_viewProjection(root_skeleton=root_skeleton,  root_list=path_list, phase='val',cam='1,2', T=T, target_view='view_1',
    #                           project_view='view_2', test_view='view_3')
    #
    # valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)

    testSet = NUCLA_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='test', cam='1,2', T=T,
                                  target_view='view_1',project_view='view_2', test_view='view_3')
    #
    testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=num_workers)


#
elif dataset == 'NTU-D':

    path_list = '/data/Yuexi/NTU-RGBD/list'
    root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    T = 20
    num_actions = 60
    trainSet = NTURGBD_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='train', T=T, target_view='C001', project_view='C002')
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    valSet = NTURGBD_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='val', T=T,target_view='C001', project_view='C002')

    valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)
    


'initialized action predictor'
N = 2 * 40 ############ modify?!!
P, Pall = gridRing(N)
Drr = abs(P)
DrrHeat = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
DthetaHeat = torch.from_numpy(Dtheta).float()

net = detectAction(classNum=num_actions, DrrImg=DrrImg,DthetaImg=DthetaImg, DrrHeat=DrrHeat, DthetaHeat=DthetaHeat, 
                    T=T, outRes=64, kinetics_pretrain= None, gpu_id=gpu_id)

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

'load pre-trained model'
# # KFPN
# K_FPN = keyframeProposalNet(numFrame=40, Drr=DrrImg, Dtheta=DthetaImg, gpu_id=gpu_id, backbone='Resnet18')
# newDict = K_FPN.state_dict()
# pre_dict = {k: v for k, v in state_dict.items() if k in newDict}
# newDict.update(pre_dict)
# K_FPN.load_state_dict(newDict)
# net.K_FPN = load_preTrained_model(K_FPN, net.K_FPN)
# I3D
kinetics_pretrain = os.getcwd() + '/pretrained/i3d_kinetics.pth'
BaseNet = BaseNet('i3d', 'rgb', kinetics_pretrain)
i3d_conv = build_conv('i3d', kinetics_pretrain, 'global', freeze_affine=True)
net.actionPredictorI3D.featureExtractor = load_preTrained_model(BaseNet, net.actionPredictorI3D.featureExtractor)
net.actionPredictorI3D.i3d_conv = load_preTrained_model(i3d_conv, net.actionPredictorI3D.i3d_conv)
# GCN
# gcn_pretrain = os.getcwd() + '/pretrained/st_gcn.kinetics-6fa43f73.pth'
# gcn_dict = torch.load(gcn_pretrain)
# new_gcn_dict = net.actionPredictorGCN.state_dict()
# dicts = {k: v for k, v in gcn_dict.items() if k in new_gcn_dict}
# new_gcn_dict.update(dicts)
# net.actionPredictorGCN.load_state_dict(new_gcn_dict)
# for param in net.actionPredictorGCN.parameters():
#     param.requires_grad = False

# net.actionPredictorGCN.CLS.weight.requires_grad=True

net.cuda(gpu_id)

################### Optimization and Criterion#################
alpha = 3

Epoch = 150

lr_heat, lr_i3d, lr_gcn = 1e-3, 1e-5, 1e-3
# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.0001, momentum=0.9)
# optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.actionPredictorHeat.parameters()), 'lr':lr_heat},
#                              {'params':filter(lambda x: x.requires_grad, net.actionPredictorI3D.parameters()), 'lr':lr_i3d},
#                              {'params':filter(lambda x: x.requires_grad, net.actionPredictorGCN.parameters()), 'lr':lr_gcn}],
#                             weight_decay=0.0001, momentum=0.9)

optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.actionPredictorHeat.parameters()), 'lr':lr_heat},
                             {'params':filter(lambda x: x.requires_grad, net.actionPredictorI3D.parameters()), 'lr':lr_i3d}],
                            weight_decay=0.0001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
criterion = nn.CrossEntropyLoss()
######################## Main #################################
stream = 'i3d+heat'
print(f'start training: heat, lr={lr_i3d}, stream={stream}')
for epoch in range(0, Epoch+1):
    lossVal = []
    loss_I3D = []
    loss_Heat = []
    loss_fusion = []
    print('training epoch:', epoch)
    # net.train()
    pbar = tqdm(total=len(trainloader),file=sys.stdout)
    num_display = int(len(trainloader)/5)
    for i, sample in enumerate(trainloader):
        optimizer.zero_grad()
        img_data = sample['imgSequence_to_use']
        skeleton = sample['sequence_to_use']
        heatmaps = sample['heatmap_to_use']
        label = sample['actLabel'].cuda(gpu_id)
        if T==0:
            nframe = sample['info']['T_sample'][0]
            # T_heat = 
        else:
            nframe = sample['nframes']

        inputImgData = img_data[0].cuda(gpu_id).float()
        inputHeatmaps = heatmaps.cuda(gpu_id).reshape(heatmaps.shape[0], nframe, -1).float()
        inputPoseData = skeleton.permute(0,3,1,2).unsqueeze(4).cuda(gpu_id).float()

        # print('sample id:', i, 'img shape:', inputImgData.shape, 'pose shape:', inputPoseData.shape)
        # Len = inputImgData.shape[0]
        seqLen = inputImgData.shape[0]
        if seqLen == nframe.item():
            Len = seqLen
        elif seqLen > nframe.item():
            Len = nframe.item()
        else:
            Len = seqLen
        pred,_ = net(inputImgData, alpha, inputHeatmaps, inputPoseData,Len)

        # loss = criterion(pred['label_i3d'], label)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        # print('check')
        lossVal.append(loss.data.item())
        # loss_I3D.append(loss_i3d.item())
        # loss_Heat.append(loss_heat.item())
        # loss_fusion.append(criterion(pred['label_heat'], out_i3d).item())
        if ((i+1) % num_display == 0):    
            pbar.update(num_display)
        elif (i == len(trainloader)-1):    
            pbar.update(len(valloader)%num_display)
    pbar.close()
    # if epoch == Epoch:
    if epoch % 5 == 0:
        print(f'validating {stream} stream:')
        count = 0
        keyFrames = []
        # totFrames = 0
        with torch.no_grad():
            # net_val = net.eval()
            pbar = tqdm(total=len(valloader),file=sys.stdout)
            num_display=int(len(valloader)/5)
            for i, sample in enumerate(valloader):
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

                # Len = inputImgData.shape[0]
                seqLen = inputImgData.shape[0]
                if seqLen == nframe.item():
                    Len = seqLen
                elif seqLen > nframe.item():
                    Len = nframe.item()
                else:
                    Len = seqLen
                pred_dict, _ = net(inputImgData, alpha, inputHeatmaps, inputPoseData, Len)

                # pred = 0.5*(pred_dict['label_heat'] + pred_dict['label_i3d'])
                # pred = pred_dict['label_i3d']
                pred = pred_dict
                if label.item() == torch.argmax(pred).cpu().data.item():
                    count = count + 1

                # keyFrames.append(numKey)
                # totFrames = totFrames + nframe
                # print('sample id:', i, 'label:', label.item(), 'out:',
                #         torch.argmax(pred).cpu().data.item())
                if ((i+1) % num_display == 0):    
                    pbar.update(num_display)
                elif (i == len(valloader)-1):    
                    pbar.update(len(valloader)%num_display)
            pbar.close()
            print('total sample:', valSet.__len__(), 'correct pred:', count, 'acc:', count/valSet.__len__())

    loss_val = np.mean(np.array(lossVal))
    print('Epoch:', epoch, '|loss:', loss_val)
    # print('Epoch:', epoch, '|loss:', loss_val, '|loss_i3d:', np.mean(np.array(loss_I3D)), '|loss_heat:', np.mean(np.array(loss_Heat)),
    #       '|loss_fusion:', np.mean(np.array(loss_fusion)))

    scheduler.step()
    # if epoch % 10 == 0:
        # torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
        #             'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

print('done')