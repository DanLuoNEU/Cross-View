from scipy.spatial import distance
import scipy.io
import torch.nn
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import binaryCoding
from dataset.SytheticData import *
from lossFunction import hashingLoss

gpu_id = 3
Npole = 161
dataset = 'Sythetic'
net = binaryCoding(num_binary=Npole).cuda(gpu_id)

modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'

modelPath = modelRoot + dataset + '/BiSythetic_BiCE_m8Alpha2_11_6poles/'
stateDict = torch.load(os.path.join(modelPath, '90.pth'))['state_dict']
net.load_state_dict(stateDict)
net.cuda(gpu_id)
net.eval()

num_Sample = 200
view1Data, view2Data = generateData(num_Sample, Npole)

testSet = generateSytheticData(view1Data=view1Data, view2Data=view2Data, Npole=Npole, phase='test')
testloader = data.DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
print('# test:', testSet.__len__())

with torch.no_grad():
    for i, sample in enumerate(testloader):
        view1 = sample['view1'].cuda(gpu_id)
        view2 = sample['view2'].cuda(gpu_id)
        y = sample['cls'].data.item()

        b1 = net(view1)
        b2 = net(view2)

        b1[b1>0] = 1
        b1[b1<0] = -1

        b2[b2 > 0] = 1
        b2[b2 < 0] = -1

        out_b1 = b1[0].cpu().numpy().tolist()
        out_b2 = b2[0].cpu().numpy().tolist()
        dist = distance.hamming(out_b1, out_b2)
        print('cls:', y, 'dist:', dist)

        # if y == 1:
        #     data = {'b1': b1.cpu().numpy(), 'b2': b2.cpu().numpy(), 'c1': view1.cpu().numpy(), 'c2': view2.cpu().numpy()}
        #     file_name = './matFile/' + dataset + '_BiCE_m8A2_11_ep90_' + 'cls'+ str(y) + '_6poles.mat'
        #     scipy.io.savemat(file_name, mdict=data)
        #
        #     print('check')

print('done')
