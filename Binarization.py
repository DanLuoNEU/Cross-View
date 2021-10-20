from torch.utils.data import DataLoader
from utils import *
import scipy.io
from modelZoo.networks import *
from scipy.spatial import distance
import torch.nn
from modelZoo.sparseCoding import *
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from modelZoo.DyanOF import *
from dataset.NTU_viewProjection import *
from torch.optim import lr_scheduler
from modelZoo.gumbel_module import *
from modelZoo.BinaryCoding import *
from lossFunction import hashingLoss, CrossEntropyLoss

gpu_id = 1
num_workers = 4

dataset = 'NUCLA'
N = 40*2
Epoch = 30
dataType = '2D'
# clip = 'Single'
clip = 'Multi'
fusion = False
lam1 = 1
lam2 = 1
LR = 1e-4
P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

# Encoder = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)
# Encoder.cuda(gpu_id)
# Encoder = DyanEncoder(Drr, Dtheta,0.01, gpu_id)
# Encoder.cuda((gpu_id))




# for m in Encoder.parameters():
#     m.requires_grad = False
#
# net = binaryCoding(num_binary=N+1).cuda(gpu_id)

net = binarizeSparseCode(num_binary=2*N+1, Drr=Drr, Dtheta=Dtheta, Inference=True, gpu_id=gpu_id)
net.cuda(gpu_id)
# for m in net.sparseCoding.parameters():
#     m.requires_grad = False

modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
# saveModel = os.path.join(modelRoot, dataset, '/BinarizeSparseCode_m32A1')
saveModel = modelRoot + dataset + '/newDYAN/BinarizeSparseCode_hardGumbel_v2/'
if not os.path.exists(saveModel):
    os.makedirs(saveModel)

T = 36
# njt = 20
# dim = 2
# MARGIN = 64
# Alpha = 1

if dataset == 'NUCLA':
    num_class = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_3d'
    trainSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='train', cam='2,1', T=T,
                                   target_view='view_2', project_view='view_1', test_view='view_3')
    # #
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=num_workers)

    valSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T,
                                  target_view='view_2',
                                  project_view='view_1', test_view='view_3')
    valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)


# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001, momentum=0.9)
optimizer = torch.optim.SGD( net.parameters(), lr=LR, weight_decay=0.001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 130], gamma=0.1)
LOSS = []
ACC = []
mseLoss = torch.nn.MSELoss()
# print('parameters: m=5, alpha=1, hash:CE = 1:2, add sparse')
print('lam1:', lam1, 'lam2:', lam2, 'LR:', LR)
for epoch in range(0, Epoch+1):
    count_cls0 = 0
    count_cls1 = 0
    print('start training epoch:', epoch)
    lossVal = []
    # lossHash = []
    # lossL1 = []
    # lossL3 = []
    # lossL2 = []
    # lossCE1 = []
    # lossCE2 = []
    lossBi = []
    lossMSE = []
    for i, sample in enumerate(trainloader):
        optimizer.zero_grad()

        input_v1 = sample['target_skeleton'].float().cuda(gpu_id)
        input_v2 = sample['project_skeleton'].float().cuda(gpu_id)

        t1 = input_v1.shape[2]
        t2 = input_v2.shape[2]

        clipBI1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
        clipBI2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)
        clipMSE1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
        clipMSE2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)
        for clip in range(0, input_v2.shape[1]):

            v1_clip = input_v1[:,clip,:,:,:].reshape(1,t1,-1)
            v2_clip = input_v2[:,clip,:,:,:].reshape(1,t2, -1)

            b1, c1, dict1 = net(v1_clip,t1)
            b2, c2, dict2 = net(v2_clip,t2)
            # clipBI1[clip] = torch.sum(b1)/((2*N+1)*50)
            clipBI1[clip] = torch.norm(b1)
            outClip_v1 = torch.matmul( dict1, c1*b1)
            clipMSE1[clip] = mseLoss(outClip_v1, v1_clip)

            # clipBI2[clip] = torch.sum(b2)/((2*N+1)*50)
            clipBI2[clip] = torch.norm(b2)
            outClip_v2 = torch.matmul(dict2, c2*b2)
            clipMSE2[clip] = mseLoss(outClip_v2, v2_clip)
            # print('check')


        loss1 = lam1* (torch.mean(clipBI1)) + lam2 * (torch.mean(clipMSE1))
        loss2 = lam1 * (torch.mean(clipBI2)) + lam2 * (torch.mean(clipMSE2))

        loss = loss1 + loss2
        loss.backward()
        # print('rr.grad:', net.sparseCoding.rr.grad, 'theta.grad:', net.sparseCoding.theta.grad)
        optimizer.step()
        lossVal.append(loss.data.item())
        lossBi.append((torch.mean(clipBI1)).data.item() + (torch.mean(clipBI2)).data.item())
        lossMSE.append((torch.mean(clipMSE1)).data.item() + (torch.mean(clipMSE2)).data.item())
    loss_val = np.mean(np.array(lossVal))
    print('epoch:', epoch, '|loss:', loss_val,  '|Bi:', np.mean(np.array(lossBi)),
          '|mse:', np.mean(np.array(lossMSE)), 'sum b1:', torch.sum(b1).data.item() )
    print('b1:', b1[:,:,2], 'c1:', c1[:,:,2])


    if epoch % 2 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')


# data = {'LOSS': np.asarray(LOSS), 'Acc': np.asarray(ACC)}
# scipy.io.savemat('./matFile/BinariedCoding_m32A1.mat', mdict=data)
print('done')




