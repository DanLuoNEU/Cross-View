from torch.optim import lr_scheduler
import torch.utils.data as dataset
from torch.utils.data import DataLoader
from modelZoo.gumbel_module import *
from dataset.SytheticData import *
from modelZoo.BinaryCoding import *
from utils import *
import scipy.io
random.seed(0)
gpu_id = 1
import pdb
Epoch = 10
N = 15*2
LR = 1e-6
lam1 = 0 # BI
lam2 = 1  # MSE
T = 36
dataSet = 'Sythetic'
modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
saveModel = modelRoot + dataSet + '/hardGumbel/'
# if not os.path.exists(saveModel):
#     os.makedirs(saveModel)

num_Sample = 1000
P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

# dictionary = creatRealDictionary(T,Drr,Dtheta,gpu_id)
dictionary = scipy.io.loadmat('./matFile_new/syntheticDictionary.mat')['dictionary']
dictionary = torch.FloatTensor(dictionary)

# nonZeroID = [1, 45, 64, 87, 96, 102, 145, 156]
nonZeroID = [1, 29, 21, 44, 30, 48, 50, 60]
nonZeroValue = torch.randn((num_Sample, len(nonZeroID), 1))
# dictionary = torch.randn(T, 2*N+1)
trainSet = generatedSyntheticData_Gumbel(dictionary=dictionary, nonZeroID=nonZeroID,nonZeroValue=nonZeroValue, Npole=2*N+1, num_sample=num_Sample,
                                         phase='train')
trainloader = dataset.DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
net = binarizeSparseCode(num_binary=2*N+1, Drr=Drr, Dtheta=Dtheta, Inference=True, gpu_id=gpu_id)
net.cuda(gpu_id)

optimizer = torch.optim.SGD( net.parameters(), lr=LR, weight_decay=0.001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
LOSS = []
ACC = []

mseLoss = torch.nn.MSELoss()
ceLoss = torch.nn.CrossEntropyLoss()

print('lam1:', lam1, 'lam2:', lam2, 'LR:',LR)
for epoch in range(0, Epoch):
    print('start training epoch:', epoch)
    lossVal = []

    lossBi = []
    lossMSE = []

    for i, sample in enumerate(trainloader):
        optimizer.zero_grad()
        inputData = sample['input'].cuda(gpu_id)  # NxTx1
        GTCoeff = sample['coeff']
        GTbi = sample['binary'].cuda(gpu_id)
        # binaryCode, Coeff, Dict = net(inputData,T)
        Coeff, Dict = net.sparseCoding(inputData, T)
        # output = torch.matmul(Dict, Coeff*binaryCode)
        output = torch.matmul(Dict, Coeff)

        # loss = lam1 * torch.sum(binaryCode)/36 + lam2 * mseLoss(inputData, output)
        loss = mseLoss(inputData, output)
        # pdb.set_trace()
        # loss = lam1 * mseLoss(GTbi, binaryCode) + lam2*mseLoss(inputData, output)

        # print('bi:', binaryCode, 'c:', Coeff)
        # print('input:', inputData[0].t())
        # print('output:', output[0].t())
        loss.backward()
        # print('rr.grad:', net.sparseCoding.rr.grad, 'thetha.grad:', net.sparseCoding.theta.grad)

        optimizer.step()
        # pdb.set_trace()

        lossVal.append(loss.data.item())
        # lossBi.append((1/36)*torch.sum(binaryCode).data.item())
        lossMSE.append(mseLoss(inputData, output).data.item())

    print('gt:', GTCoeff[0].t())
    print('pred:', Coeff[0].t())
    # print('binary:', binaryCode[0].t())
    print('gt binary:', GTbi[0].t())
    print('input:', inputData[0].t())
    print('output:', output[0].t())

    # print('sum(bi):', torch.sum(binaryCode))
    loss_val = np.mean(np.asarray(lossVal))
    # loss_bi = np.mean(np.asarray(lossBi))
    loss_mse = np.mean(np.asarray(lossMSE))
    # pdb.set_trace()

    # print('epoch:', epoch, 'loss:', loss_val, 'bi:', loss_bi, 'mse:', loss_mse)
    print('epoch:', epoch, 'loss:', loss_val)
    scheduler.step()

    # if epoch % 10 == 0:
    #     torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
    #             'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

print('done')
