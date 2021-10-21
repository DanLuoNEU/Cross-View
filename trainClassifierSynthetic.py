from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import binaryCoding,classificationHead
import torch.nn
from dataset.SytheticData import *
import scipy.io
from lossFunction import *

gpu_id = 1
Npole = 161
num_Sample = 300
Epoch = 50
njt = 18

dataset = 'Sythetic'
dataType = '2D'
getBinaryCode = binaryCoding(num_binary=Npole).cuda(gpu_id)

modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'

# modelPath = modelRoot + dataset + '/BiSythetic_BiOnly_m8Alpha01_10_6poles/'
# stateDict = torch.load(os.path.join(modelPath, '150.pth'))['state_dict']
# getBinaryCode.load_state_dict(stateDict)
# getBinaryCode.cuda(gpu_id)
# getBinaryCode.eval()

net = classificationHead(num_class=2, Npole=161, dataType=dataType).cuda(gpu_id)
MultiClassData = getMultiClassData(num_Sample, 161, 36)
# view1Data, view2Data = generateData(num_Sample, Npole)
#
# trainSet = generateSytheticData(view1Data=view1Data, view2Data=view2Data, Npole=Npole, phase='train')

trainSet = generateSytheticData_MultiClass(data=MultiClassData, num_Sample=num_Sample, Npole=Npole, dim=njt*2, phase='train')
trainloader = data.DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

# valSet = generateSytheticData(view1Data=view1Data, view2Data=view2Data, Npole=Npole, phase='val')
valSet = generateSytheticData_MultiClass(data= MultiClassData,num_Sample=num_Sample, Npole=Npole, dim=njt*2, phase='val')
valloader = data.DataLoader(valSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-4, weight_decay=0.0001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

Criterion = torch.nn.CrossEntropyLoss()
LOSS = []
ACC = []
for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    lossVal = []
    for i, sample in enumerate(trainloader):
        # print('sample:', i)
        optimizer.zero_grad()
        # view1 = sample['view1'].cuda(gpu_id)
        # view2 = sample['view2'].cuda(gpu_id)

        y = sample['class'].cuda(gpu_id)
        # print(sample['class'])
        input = sample['input'].float().cuda(gpu_id).reshape(1, Npole, njt, 2)
        # input = getBinaryCode(sample['input'].cuda(gpu_id)).unsqueeze(2).unsqueeze(3)
        # input = torch.cat((view1, view2),1)


        # b1 = getBinaryCode(view1).unsqueeze(2).unsqueeze(3)
        # b2 = getBinaryCode(view2).unsqueeze(2).unsqueeze(3)
        # input = torch.cat((b1, b2),1)

        label = net(input)

        loss = Criterion(label, y)
        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())

    loss_val = np.mean(np.array(lossVal))
    LOSS.append(loss_val)
    print('epoch:', epoch, '|loss:', loss_val)
    scheduler.step()
    if epoch % 1 == 0:
        print('start validating:')
        count = 0
        with torch.no_grad():
            for i, sample in enumerate(valloader):
                # view1 = sample['view1'].cuda(gpu_id)
                # view2 = sample['view2'].cuda(gpu_id)
                cls = sample['class'].data.item()
                # input = sample['input'].cuda(gpu_id)
                # input = getBinaryCode(sample['input'].cuda(gpu_id)).unsqueeze(2).unsqueeze(3)
                input = sample['input'].float().cuda(gpu_id).reshape(1, Npole, njt, 2)
                # input = torch.cat((view1, view2), 1)
                label = net(input)

                pred = torch.argmax(label).data.item()
                if pred == cls:
                    count+= 1
            Acc = count/valSet.__len__()

            print('epoch:', epoch, 'Acc:', Acc)
            ACC.append(Acc)

data = {'LOSS':np.asarray(LOSS), 'Acc':np.asarray(ACC)}
# scipy.io.savemat('./matFile/SyntheticClassification_Coeff_5Cls.mat', mdict=data)
print('done')


    # print('check')
