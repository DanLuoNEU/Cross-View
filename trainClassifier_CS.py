from torch.utils.data import DataLoader
from utils import *
import scipy.io
from modelZoo.networks import *
from scipy.spatial import distance
import torch.nn
from modelZoo.sparseCoding import sparseCodingGenerator
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from dataset.NUCLA_viewProjection_CS import *
from dataset.NTURGBDsubject import *
from modelZoo.DyanOF import *
from dataset.NTU_viewProjection import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from lossFunction import binaryLoss
# torch.backends.cudnn.enabled = False
from lossFunction import hashingLoss, CrossEntropyLoss

gpu_id = 3
num_workers = 4
PRE = 0
T = 36
# dataset = 'NUCLA'
dataset = 'NTU'
Alpha = 0.5
lam1 = 2
lam2 = 1
N = 40*4
Epoch = 150
dataType = '2D'
clip = 'Single'
fusion = False

P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()
modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
# saveModel = os.path.join(modelRoot, dataset, '/BinarizeSparseCode_m32A1')
# saveModel = modelRoot + dataset + '/3DSkeleton/CS/single_lam1051_4LayerGN/'  #note: for 2D, 2 stream, UCLA
saveModel = modelRoot + dataset + '/singleClip/singleClip_T36_CS/'
if not os.path.exists(saveModel):
    os.makedirs(saveModel)

# modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
# modelPath = modelRoot + dataset + '/sparseCodeGenerator_NUCLA_T36_fist01_openPose/'
# stateDict = torch.load(os.path.join(modelPath, '150.pth'))['state_dict']

# Drr = stateDict['l1.rr']
# Dtheta = stateDict['l1.theta']
#
# Encoder = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)
# Encoder.cuda(gpu_id)
#
# for m in Encoder.parameters():
#     m.requires_grad = False



if dataset == 'NUCLA':
    num_class = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLAsubject(root_list=path_list, dataType=dataType, clip=clip, phase='train', T=T)
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    valSet = NUCLAsubject(root_list=path_list, dataType=dataType, clip=clip, phase='test', T=T)
    valloader = DataLoader(valSet, batch_size=1, shuffle=True, num_workers=num_workers)

else:
    num_class = 60
    # root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    nanList = list(np.load('./NTU_badList.npz')['x'])
    if dataType == '3D':
        root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    else:
        root_skeleton = "/data/NTU-RGBD/poses"

    trainSet = NTURGBDsubject(root_skeleton, nanList, dataType=dataType, clip=clip, phase='train', T=36)
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    valSet = NTURGBDsubject(root_skeleton, nanList, dataType=dataType, clip=clip, phase='test', T=36)
    valloader = torch.utils.data.DataLoader(valSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)




# net = classificationHead(num_class=num_class, Npole=(N+1)).cuda(gpu_id)
# net = classificationWBinarization(num_class=num_class, Npole=(N+1), num_binary=(N+1)).cuda(gpu_id)
# net = classificationWSparseCode(num_class=num_class, Npole=N+1, Drr=Drr, Dtheta=Dtheta, PRE=0, gpu_id=gpu_id).cuda(gpu_id)
# net = Fullclassification(num_class=10, Npole=N+1, num_binary=N+1, Drr=Drr, Dtheta=Dtheta, PRE=0,  dim=2, dataType=dataType, gpu_id=gpu_id).cuda(gpu_id)
kinetics_pretrain = './pretrained/i3d_kinetics.pth'
net = twoStreamClassification(num_class=num_class, Npole=(N+1), num_binary=(N+1), Drr=Drr, Dtheta=Dtheta,
                                  PRE=0, dim=2, gpu_id=gpu_id, dataType=dataType, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

net.train()
# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.0001, momentum=0.9)
optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.dynamicsClassifier.parameters()), 'lr':1e-3},
{'params':filter(lambda x: x.requires_grad, net.RGBClassifier.parameters()), 'lr':1e-4}], weight_decay=0.001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 130], gamma=0.1)
Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()

# Criterion = torch.nn.BCELoss()
LOSS = []
ACC = []
# print('cls:bi:reconst=2:0.3:1')
for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    lossVal = []
    lossCls = []
    lossBi = []
    lossMSE = []
    for i, sample in enumerate(trainloader):

        # print('sample:', i)
        optimizer.zero_grad()


        'single clip'
        skeleton = sample['input_skeleton']['normSkeleton'].float().cuda(gpu_id)
        imageData = sample['input_image'].float().cuda(gpu_id)
        # y = sample['cls'].cuda(gpu_id)
        y = sample['action'].cuda(gpu_id)
        # print('target video:', sample['target_info']['name_sample'], 'project video:', sample['project_info']['name_sample'])
        t = skeleton.shape[1]
        inputSkeleton = skeleton.reshape(1, t, -1)

        '2 Stream'

        label, b, outSkeleton = net(inputSkeleton, imageData, t, fusion)

        loss = lam1 * Criterion(label, y) + Alpha * binaryLoss(b, gpu_id) + lam2 * mseLoss(outSkeleton, inputSkeleton)

        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())
        lossCls.append(Criterion(label, y).data.item())
        lossBi.append(binaryLoss(b, gpu_id).data.item())
        lossMSE.append(mseLoss(outSkeleton, inputSkeleton).data.item())

        """""
        c1, _ = Encoder.forward2(input_v1, T)
        c2, _ = Encoder.forward2(input_v2, T)

        c1 = c1.reshape(1, N+1, int(input_v1.shape[-1]/2), 2)
        c2 = c2.reshape(1, N+1, int(input_v2.shape[-1]/2), 2)
        """

        # label = net(input)
        'only using classification net'
        """""
        label1 = net(c1)
        label2= net(c2)

        loss1 = Criterion(label1, y)
        loss2 = Criterion(label2, y)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())
        """

        """""
        'classifier + DYAN'

        label, outputSkeleton = net(inputSkeleton, T)

        loss = lam1 * Criterion(label, y) + lam2 * mseLoss(outputSkeleton, inputSkeleton)

        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())
        lossCls.append(Criterion(label, y).data.item())
        lossMSE.append(mseLoss(outputSkeleton, inputSkeleton).data.item())
        
        'classifier + binary code'
        
        label1, b1 = net(c1)
        label2, b2 = net(c2)
        loss1 = lam1*Criterion(label1, y) + Alpha*binaryLoss(b1, gpu_id)
        loss2 = lam1*Criterion(label2, y) + Alpha*binaryLoss(b2,gpu_id)
        loss = loss1 + loss2
        
        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())
        lossCls.append(Criterion(label1, y).data.item()+Criterion(label2, y).data.item())
        lossBi.append(Alpha*binaryLoss(b1, gpu_id).data.item() + Alpha*binaryLoss(b2,gpu_id).data.item())
        
        
        'DYAN + Binarization + Classifier'
        label, b, outSkeleton = net(inputSkeleton, t)

        loss = lam1*Criterion(label, y) + Alpha * binaryLoss(b, gpu_id) + lam2*mseLoss(outSkeleton, inputSkeleton)


        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())
        lossCls.append(Criterion(label, y).data.item() )
        lossBi.append(binaryLoss(b, gpu_id).data.item())
        lossMSE.append(mseLoss(outSkeleton, inputSkeleton).data.item())
        """

    loss_val = np.mean(np.array(lossVal))
    LOSS.append(loss_val)
    # print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|mse:', np.mean(np.array(lossMSE)))
    # print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|Bi:', np.mean(np.array(lossBi)))
    print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|Bi:', np.mean(np.array(lossBi)),
          '|mse:', np.mean(np.array(lossMSE)))
    # print('epoch:', epoch, '|loss:', loss_val)
    scheduler.step()
    if epoch % 5 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

    if epoch % 5 == 0:
        print('start validating:')
        count = 0
        pred_cnt = 0
        Acc = []
        with torch.no_grad():
            for i, sample in enumerate(valloader):
                # skeleton_v1 = sample['target_view_skeleton']['normSkeleton'].cuda(gpu_id)
                # skeleton_v2 = sample['project_view_skeleton']['normSkeleton'].cuda(gpu_id)
                skeleton = sample['input_skeleton']['normSkeleton'].float().cuda(gpu_id)
                imageData = sample['input_image'].float().cuda(gpu_id)

                t = skeleton.shape[1]
                y = sample['action'].data.item()
                # y = sample['cls'].data.item()
                input = skeleton.reshape(1, t, -1)
                label, _ , _ = net(input, imageData, t, fusion)  # 2 stream
                # label, _, _ = net(input, t) # DY+BL+CL


                # c, _ = Encoder.forward2(input, T)
                # c = c.reshape(1, N + 1, int(input.shape[-1]/2), 2)
                # label = net(c)  # CL only
                # label, _ = net(c) # 'BI + CL'

                # label, _ = net(input, T) # 'DY + CL'

                pred = torch.argmax(label).data.item()
                # print('sample:',i, 'pred:', pred, 'gt:', y)
                count += 1
                # if pred1 == y:
                #     pred_cnt +=1
                # elif pred2 == y:
                #     pred_cnt += 1
                if pred == y:
                    pred_cnt += 1

                # for n in range(0, label.shape[0]):
                #     pred = torch.argmax(label[n]).data.item()
                #     if pred == y[0]:
                #         count+= 1
                # acc = count/label.shape[0]
            # Acc.append(acc)
            # Acc = count/valSet.__len__()
            Acc = pred_cnt/count

            print('epoch:', epoch, 'Acc:%.4f'% Acc, 'count:', count, 'pred_cnt:', pred_cnt)
            ACC.append(Acc)

# data = {'LOSS':np.asarray(LOSS), 'Acc':np.asarray(ACC)}
# scipy.io.savemat('./matFile/Classifier_train_v12_test_v3_10cls.mat', mdict=data)

print('done')