# from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *

import scipy.io
# from modelZoo.networks import *
from modelZoo.sparseCoding import sparseCodingGenerator
import torch
from scipy.spatial import distance
import torch.nn
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import binaryCoding

from dataset.SytheticData import *
from lossFunction import *

gpu_id = 1

Npole = 161
Epoch = 30
MARGIN = 16
ALPHA = 0.5
N = 40*4
PRE = 0
njt = 18

dataset = 'Sythetic'
net = binaryCoding(num_binary=Npole).cuda(gpu_id)


modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
saveModel = modelRoot + dataset + '/BiSythetic_BiOnly_5cls/'
# if not os.path.exists(saveModel):
#     os.makedirs(saveModel)

num_Sample = 500
# view1Data, view2Data = generateData(num_Sample, Npole)
#
# trainSet = generateSytheticData(view1Data=view1Data, view2Data=view2Data, Npole=Npole, phase='train')

MultiClassData = getMultiClassData(num_Sample, 161, 36)
trainSet = generateSytheticData_MultiClass(data=MultiClassData, num_Sample=num_Sample, Npole=Npole, dim=36, phase='train')
trainloader = data.DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

# valSet = generateSytheticData(view1Data=view1Data, view2Data=view2Data, Npole=Npole, phase='val')
# valloader = data.DataLoader(valSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

valSet = generateSytheticData_MultiClass(data=MultiClassData, num_Sample=num_Sample, Npole=Npole, dim=36, phase='val')
valloader = data.DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 130], gamma=0.1)
print('# of train data:', trainSet.__len__(), '# of val data:', valSet.__len__())
spLoss = torch.nn.L1Loss()

# tb = SummaryWriter()
SP1 = []
SP2 = []
sameCLS = []
diffCLS = []
BiLoss = []
LOSS = []
ACC = []
for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    lossVal = []
    lossL1 = []
    lossL3 = []
    lossL2 = []
    lossHash = []
    lossSP1 = []
    lossSP2 = []
    lossCE1 = []
    lossCE2 = []
    for i, sample in enumerate(trainloader):
        # print('sample:', i)
        optimizer.zero_grad()
        view1 = sample['view1'].float().cuda(gpu_id).reshape(1, N+1, njt,2)
        view2 = sample['view2'].float().cuda(gpu_id).reshape(1, N+1, njt,2)
        # print(view1.shape)
        y = sample['class'].data.item()
        # y = sample['class'].cuda(gpu_id)
        # input = sample['input'].cuda(gpu_id)

        b1 = net(view1)
        b2 = net(view2)

        hashingloss, l1, l2, l3 = hashingLoss(b1, b2, y=y, m=MARGIN, alpha=ALPHA, gpu_id=gpu_id)



        sp1 = torch.norm(b1 + 1, p=1)
        sp2 = torch.norm(b2 + 1, p=1)
        # const = torch.Tensor(12).cuda(gpu_id)
        # sp1 = spLoss(b1+1, const)
        # sp2 = spLoss(b2+1, const)
        ce1 = CrossEntropyLoss(view1[:,:,0,0], b1, gpu_id)
        ce2 = CrossEntropyLoss(view2[:,:,0,0], b2, gpu_id)


        # loss = hashingloss + 1*(sp1 + sp2) + 1*(ce1+ce2)
        loss = hashingloss

        loss.backward()
        # print(net.conv[0].weight.grad)
        optimizer.step()
        lossVal.append(loss.data.item())
        lossHash.append(hashingloss.data.item())
        lossL1.append(l1.data.item())
        lossL2.append(l2.data.item())
        lossL3.append(l3.data.item())
        lossSP1.append(sp1.data.item())
        lossSP2.append(sp2.data.item())

        lossCE1.append(ce1.data.item())
        lossCE2.append(ce2.data.item())


    loss_val = np.mean(np.array(lossVal))
    loss_hash = np.mean((np.array(lossHash)))
    loss_l1 = np.mean(np.array(lossL1))
    loss_l2 = np.mean(np.array(lossL2))
    loss_l3 = np.mean(np.array(lossL3))
    loss_sp1 = np.mean(np.array(lossSP1))
    loss_sp2 = np.mean(np.array(lossSP2))
    loss_ce1 = np.mean(np.array(lossCE1))
    loss_ce2 = np.mean(np.array(lossCE2))
    print('epoch:', epoch, '|tot loss:', loss_val, '|similarity term:', loss_l1,'|dis-similarity:', loss_l2,
          '|binary term:', loss_l3)
    # print('epoch:', epoch, '|sp1:', loss_sp1, '|sp2:', loss_sp2, '|ce1:', loss_ce1, '|ce2:', loss_ce2)
    # print('epoch:', epoch, '|tot loss:', loss_val)

    scheduler.step()
    # tb.add_scalar('tot loss', loss_val, epoch)

    SP1.append(loss_sp1)
    SP2.append(loss_sp2)
    sameCLS.append(loss_l1)
    diffCLS.append(loss_l2)
    BiLoss.append(loss_l3)
    LOSS.append(loss_val)

    # if epoch % 10 == 0:
    #     torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
    #                 'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

    if epoch % 1 == 0:
        print('validating...')
        count = 0

        # Error = []
        with torch.no_grad():
            hamingDIS_0 = []
            hamingDIS_1 = []
            for i, sample in enumerate(valloader):
                view1 = sample['view1'].float().cuda(gpu_id).reshape(1, N+1, njt,2)
                view2 = sample['view2'].float().cuda(gpu_id).reshape(1, N+1, njt,2)
                cls = sample['class'].data.item()

                b1 = net(view1)
                b2 = net(view2)

                b1[b1 > 0] = 1
                b1[b1 < 0] = -1

                b2[b2 > 0] = 1
                b2[b2 < 0] = -1

                out_b1 = b1[0].cpu().numpy().tolist()
                out_b2 = b2[0].cpu().numpy().tolist()
                dist = distance.hamming(out_b1, out_b2)

                if dist == 0:
                    label = 0
                else:
                    label = 1

                if label == cls:
                    count +=1
                # if cls == 0:
                #     hamingDIS_0.append(dist)
                # else:
                #     hamingDIS_1.append(dist)

            # print('epoch:', epoch, 'hamingDIS_0:', np.mean(np.asarray(hamingDIS_0)), 'hamingDIS_1:', np.mean(np.asarray(hamingDIS_1)))

            acc = count/valSet.__len__()
            ACC.append(acc)
            print('epoch:', epoch, 'ACC:', np.mean(np.asarray(ACC)), 'pred:', label, 'gt:', cls)
            # print('max output:',torch.max(b1), 'min output:', torch.min(b1))


# data = {'SP1':np.asarray(SP1), 'SP2':np.asarray(SP2), 'sampleCLS':np.asarray(sameCLS), 'diffCLS':np.asarray(diffCLS), 'BiLoss':np.asarray(BiLoss)}
# scipy.io.savemat('./matFile/synthetic_BiCE_6poles_m8A211_loss.mat', mdict=data)
data = {'LOSS': np.asarray(LOSS), 'Acc': np.asarray(ACC)}
# scipy.io.savemat('./matFile/syntheticClassification_Bicode.mat', mdict=data)


print('done')
# tb.close()