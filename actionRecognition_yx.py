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
# from dataset.NUCLAskeleton import *

# import cv2
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False

##### Configuration #####
# GPU
gpu_id = 2
num_workers = 4# 4
# Experiment
T = 0  #8 #40
# List Roots
## NTU-RGBD dataset
# path_list = f"/data/Dan/NTU-RGBD/list/fast_25/T{T}"
## Northwestern-UCLA dataset
path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
# Model Paths
# modelFolder = '/home/yuexi/Documents/keyFrameModel/RealData/JHMDB/resnet18'
# saveModel = '/home/dan/Projects/CrossView/exp/I3D_heat_fusion_wPre/'
# if not os.path.exists(saveModel):
#     os.makedirs(saveModel)
# DataLoader
if 'NTU-RGBD' in path_list:
    from dataset.NTURGBDskeleton import NTURGBDskeleton

    numJoint = 25
    num_actions = 60

    trainSet = NTURGBDskeleton(root_list=path_list, phase='train', cam='1', T=T)
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)
    valSet = NTURGBDskeleton(root_list=path_list, phase='val',cam='1', T=T)
    valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)
elif 'N-UCLA' in path_list:
    from dataset.NUCLAskeleton import NUCLAskeleton
    from dataset.NUCLAsubject import *
    numJoint = 20
    num_actions =10
    #
    # trainSet = NUCLAskeleton(root_list=path_list, phase='train', cam='1', T=0)
    # trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)
    # valSet = NUCLAskeleton(root_list=path_list, phase='test',cam='3', T=0)
    # valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)

    trainSet = NUCLAsubject(phase='train')
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)
    valSet = NUCLAsubject(phase='test')
    valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)
    

'loading pretrained kfpn'
# modelFile = os.path.join(modelFolder, 'lam41_83.pth')
# state_dict = torch.load(modelFile)['state_dict']
# DrrImg = state_dict['Drr']
# DthetaImg = state_dict['Dtheta']

'initialized action predictor'
N = 2 * 40 ############ modify?!!
P, Pall = gridRing(N)
Drr = abs(P)
DrrHeat = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
DthetaHeat = torch.from_numpy(Dtheta).float()
net = heatAction(classNum=num_actions, Drr=DrrHeat, Dtheta=DthetaHeat, PRE=0, outRes=64,backbone='Resnet18', gpu_id=gpu_id)

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


resnet = models.resnet18(pretrained=True, progress=False)
net.imageFeatExtractor.modifiedResnet = load_preTrained_model(resnet, net.imageFeatExtractor.modifiedResnet)

net.cuda(gpu_id)

################### Optimization and Criterion#################
alpha = 3
Epoch = 150

lr_heat = 1e-3
optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=lr_heat, weight_decay=0.0001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
criterion = nn.CrossEntropyLoss()
######################## Main #################################
useStream = 'heat'
print(f'start training: Stream={useStream}, lr={lr_heat}, not binarized')
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
        # inputPoseData = skeleton.permute(0,3,1,2).unsqueeze(4).cuda(gpu_id).float()

        # print('sample id:', i, 'img shape:', inputImgData.shape, 'pose shape:', inputPoseData.shape)
        # pred,_ = net(inputImgData, alpha, inputHeatmaps, inputPoseData, nframe)
        # pred = net(inputHeatmaps, nframe.item())
        seqLen = inputImgData.shape[0]
        if seqLen == nframe.item():
            Len = seqLen
        elif seqLen > nframe.item():
            Len = nframe.item()
        else:
            Len = seqLen
        pred = net.forward(inputHeatmaps[:, 0:Len], inputImgData[0:Len], Len, useStream=useStream)

        loss = criterion(pred, label)

        # loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        # print('check')
        lossVal.append(loss.data.item())

        if ((i+1) % num_display == 0):    
            pbar.update(num_display)
        elif (i == len(trainloader)-1):    
            pbar.update(len(valloader)%num_display)
    pbar.close()
    # if epoch == Epoch:
    if epoch % 5 == 0:
        print(f'start training: Stream={useStream}')
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

                # pred = net(inputHeatmaps, nframe)
                seqLen = inputImgData.shape[0]
                if seqLen == nframe.item():
                    Len = seqLen
                elif seqLen > nframe.item():
                    Len = nframe.item()
                else:
                    Len = seqLen
                pred = net.forward(inputHeatmaps[:, 0:Len, :], inputImgData[0:Len], Len, useStream=useStream)

                # pred_dict, numKey = net(inputImgData, alpha, inputHeatmaps, inputPoseData, nframe)

                # pred = 0.5*(pred_dict['label_heat'] + pred_dict['label_i3d'])
                # pred = pred_dict['label_heat']
                # pred = pred_dict
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