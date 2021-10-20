import numpy as np
import torch
import torchvision.models as models
from modelZoo import DyanOF
from modelZoo.networks import keyframeProposalNet
from modelZoo.sparseCoding import sparseCodingGenerator
from torch.utils.data import DataLoader, Dataset
from modelZoo.networks import load_preTrained_model

from torch.optim import lr_scheduler


from utils import *
import torch.nn as nn
import random
torch.manual_seed(1)
random.seed(1)

T = 20
FRA = T
gpu_id = 1
N = 40*4
EPOCH = 100
PRE = 0
num_workers = 2

saveFolder = '/home/yuexi/Documents/ModelFile/crossViewModel/NTU-D/heatDYAN_view12_v2/'
if not os.path.exists(saveFolder):
	os.makedirs(saveFolder)


P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

# net = OFModel(Drr, Dtheta, T, gpu_id)
net = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)

"""""
PRE = 0
net = imgFeatureDYAN(Drr, Dtheta, PRE, 'Resnet34', gpu_id)n


for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


resnet = models.resnet34(pretrained=True, progress=False)
net.imageFeatExtractor.modifiedResnet = load_preTrained_model(resnet, net.featureExtractor.modifiedResnet)
"""
net.cuda(gpu_id)

# path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
path_list = f"/data/Dan/NTU-RGBD/list/fast_25/T{T}"
if 'NTU-RGBD' in path_list:
    from dataset.NTURGBDskeleton import NTURGBDskeleton

    numJoint = 25
    num_actions = 60

    trainSet = NTURGBDskeleton(root_list=path_list, phase='train', cam='1,2', T=T)
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)
    # valSet = NTURGBDskeleton(root_list=path_list, phase='val',cam='1', T=T)
    # valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)
elif 'N-UCLA' in path_list:
    from dataset.NUCLAskeleton import NUCLAskeleton
    from dataset.NUCLAsubject import NUCLAsubject
    numJoint = 20
    num_actions =10
    #
    trainSet = NUCLAskeleton(root_list=path_list, phase='train', cam='1,2', T=0)
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)
    # valSet = NUCLAskeleton(root_list=path_list, phase='test',cam='3', T=0)
    # valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)

    # trainSet = NUCLAsubject(phase='train')
    # trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)
    # valSet = NUCLAsubject(phase='test')
    # valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)



# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-4, weight_decay=0.001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1) # if Kitti: milestones=[100,150]
loss_mse = nn.MSELoss()
start_epoch = 1
saveEvery = 5
for epoch in range(start_epoch, EPOCH+1):
    print('start training:', epoch)
    loss_value = []

    for i, sample in enumerate(trainloader):
        optimizer.zero_grad()
        img_data = sample['imgSequence_to_use']
        skeleton = sample['sequence_to_use']
        heatmaps = sample['heatmap_to_use']

        # if T == 0:
        #     nframe = sample['info']['T_sample'][0]
        #     # T_heat =
        # else:
        #     nframe = sample['nframes']

        nframe = T

        inputImgData = img_data.cuda(gpu_id).float()
        # inputPoseData = skeleton.permute(0,3,1,2).unsqueeze(4).cuda(gpu_id).float()


        # seqLen = inputImgData.shape[1]
        # if seqLen == nframe.item():
        #     Len = seqLen
        # elif seqLen > nframe.item():
        #     Len = nframe.item()
        # else:
        #     Len = seqLen
        Len = nframe

        # Len =
        targetData = heatmaps[:, 0:Len, :,:].cuda(gpu_id).reshape(heatmaps.shape[0], Len, -1).float()
        # targetData = inputImgData[:,0:Len,:,:,:].reshape(inputImgData.shape[0], Len, -1)

        outData = net(targetData, Len)

        # print(output.shape)
        loss = loss_mse(outData, targetData)

        loss.backward()
        optimizer.step()
        loss_value.append(loss.data.item())

    loss_val = np.mean(np.array(loss_value))

    print('Epoch: ', epoch, '| train loss: %.6f' % loss_val)
    scheduler.step()
    if epoch % saveEvery == 0:
        print('saving model...')
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, saveFolder + str(epoch) + '.pth')


print('done')



