from torch.utils.data import DataLoader
from utils import *
import scipy.io
from modelZoo.networks import *
from torch.autograd import Variable
from scipy.spatial import distance
import torch.nn
from modelZoo.sparseCoding import sparseCodingGenerator
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from modelZoo.DyanOF import *
from dataset.NTU_viewProjection import *
from torch.optim import lr_scheduler
from modelZoo.BinaryCoding import *
from lossFunction import binaryLoss
from dataset.NUCLA_viewProjection_CS import *
from dataset.NTURGBDsubject import *
# torch.backends.cudnn.enabled = False
from lossFunction import hashingLoss, CrossEntropyLoss

gpu_id = 3
num_workers = 2
PRE = 0
T = 36
dataset = 'NUCLA'
# dataset = 'NTU'
Alpha = 0
lam1 = 1
lam2 = 1
N = 80*2
Epoch = 80
# num_class = 10
dataType = '2D'
clip = 'Multi'
# clip = 'Single'
fusion = False

modelRoot = './crossViewModel/'
saveModel = modelRoot + dataset + '/test/2S_CS_Multi_fista03_reWeighted_noBI_NTU/'
if not os.path.exists(saveModel):
    os.makedirs(saveModel)


P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

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
    path_list = f"/data/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLAsubject(root_list=path_list, dataType=dataType, clip=clip, phase='train', T=T)
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    valSet = NUCLAsubject(root_list=path_list, dataType=dataType, clip=clip, phase='test', T=T)
    valloader = DataLoader(valSet, batch_size=1, shuffle=True, num_workers=num_workers)

else:
    num_class = 60
    # root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    # nanList = list(np.load('./NTU_badList.npz')['x'])
    with open('/data/NTU-RGBD/ntu_rgb_missings_60.txt', 'r') as f:
        nanList = f.readlines()
        nanList = [line.rstrip() for line in nanList]
    if dataType == '3D':
        root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    else:
        root_skeleton = "/data/NTU-RGBD/poses_60"

    trainSet = NTURGBDsubject(root_skeleton, nanList, dataType=dataType, clip=clip, phase='train', T=36)
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    valSet = NTURGBDsubject(root_skeleton, nanList, dataType=dataType, clip=clip, phase='test', T=36)
    valloader = torch.utils.data.DataLoader(valSet, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)



# net = classificationHead(num_class=10, Npole=(N+1)).cuda(gpu_id)
# net = classificationWBinarization(num_class=10, Npole=(N+1), num_binary=(N+1)).cuda(gpu_id)
# net = classificationWSparseCode(num_class=10, Npole=N+1, Drr=Drr, Dtheta=Dtheta, PRE=0, gpu_id=gpu_id).cuda(gpu_id)
# net = Fullclassification(num_class=num_class, Npole=N+1, num_binary=N+1, Drr=Drr, Dtheta=Dtheta, PRE=0, dim=3,dataType=dataType, gpu_id=gpu_id).cuda(gpu_id)
kinetics_pretrain = './pretrained/i3d_kinetics.pth'
net = twoStreamClassification(num_class=num_class, Npole=(N+1), num_binary=(N+1), Drr=Drr, Dtheta=Dtheta,
                            dim=2, gpu_id=gpu_id, inference=True, fistaLam=0.3, dataType=dataType, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)


net.train()

# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3, weight_decay=0.001, momentum=0.9)

optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.dynamicsClassifier.Classifier.parameters()), 'lr':1e-3},
{'params':filter(lambda x: x.requires_grad, net.dynamicsClassifier.sparseCoding.parameters()), 'lr':1e-6},
{'params':filter(lambda x: x.requires_grad, net.RGBClassifier.parameters()), 'lr':1e-4}], weight_decay=0.001, momentum=0.9)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)
Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()

# Criterion = torch.nn.BCELoss()
LOSS = []
ACC = []
# print('cls:bi:reconst=2:0.3:1')
print('alpha:', Alpha, 'lam1:', lam1, 'lam2:', lam2)
for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    lossVal = []
    lossCls = []
    lossBi = []
    lossMSE = []
    for i, sample in enumerate(trainloader):

        # print('sample:', i)
        optimizer.zero_grad()
        skeleton = sample['input_skeleton'].float().cuda(gpu_id)
        # skeleton = sample['velocity'].float().cuda(gpu_id)
        imageData = sample['input_image'].float().cuda(gpu_id)

        y = sample['action'].cuda(gpu_id)
        t = skeleton.shape[2]

        label = torch.zeros(skeleton.shape[1], num_class).cuda(gpu_id)
        label_rgb = torch.zeros(skeleton.shape[1], num_class).cuda(gpu_id)
        label_dym = torch.zeros(skeleton.shape[1], num_class).cuda(gpu_id)

        clipBI = torch.zeros(skeleton.shape[1]).cuda(gpu_id)

        clipMSE = torch.zeros(skeleton.shape[1]).cuda(gpu_id)

        for clip in range(0, skeleton.shape[1]):
            t = skeleton.shape[2]
            inputClip = skeleton[:,clip,:,:,:].reshape(1,t,-1)
            imgClip = imageData[:, clip, :, :, :]


            # label_clip, b, outClip = net(inputClip, t) # DY+BI+CL
            label_clip, b, outClip = net(inputClip,imgClip, t, fusion) #2S
            bi_gt = torch.zeros_like(b).cuda(gpu_id)
            # bi_gt2 = torch.zeros_like(b2).cuda(gpu_id)

            # loss1 = lam1 * Criterion(label1, y) + Alpha * binaryLoss(b1, gpu_id) + lam2 * mseLoss(outClip_v1, v1_clip)
            # loss2 = lam1 * Criterion(label2, y) + Alpha * binaryLoss(b2, gpu_id) + lam2 * mseLoss(outClip_v2, v2_clip)
            # clipBI[clip] = binaryLoss(b, gpu_id)
            clipBI[clip] = L1loss(b, bi_gt)
            clipMSE[clip] = mseLoss(outClip, inputClip)

            if fusion:
                label_rgb[clip] = label_clip['RGB']

                label_dym[clip] = label_clip['Dynamcis']
            else:

                label[clip] = label_clip

        if fusion:
            label_rgb = torch.mean(label_rgb, 0, keepdim=True)
            label_dym = torch.mean(label_dym, 0, keepdim=True)

            y_rgb = torch.tensor(torch.argmax(label_rgb).data.item()).cuda(gpu_id).unsqueeze(0)
            LossCl = Criterion(label_rgb, y) + Criterion(label_dym, y) + Criterion(label_dym, y_rgb)
            loss = lam1 * LossCl + Alpha * (torch.mean(clipBI)) + lam2 * (torch.mean(clipMSE))
            label = 0.5*label_rgb + 0.5*label_dym

        else:
            label = torch.mean(label, 0, keepdim=True)
            loss = lam1 * Criterion(label, y) + Alpha * (torch.mean(clipBI)) + lam2 * (torch.mean(clipMSE))
        loss.backward()
        optimizer.step()
        lossVal.append(loss.data.item())
        lossCls.append(Criterion(label, y).data.item())
        lossBi.append((torch.mean(clipBI)).data.item())
        lossMSE.append((torch.mean(clipMSE)).data.item())

    loss_val = np.mean(np.array(lossVal))
    LOSS.append(loss_val)
    # print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|mse:', np.mean(np.array(lossMSE)))
    # print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|Bi:', np.mean(np.array(lossBi)))
    print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|Bi:', np.mean(np.array(lossBi)),
          '|mse:', np.mean(np.array(lossMSE)))

    if epoch % 5 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

    scheduler.step()
    if epoch % 5 == 0:
        print('start validating:')
        count = 0
        pred_cnt = 0
        Acc = []
        with torch.no_grad():
            for i, sample in enumerate(valloader):

                skeleton = sample['input_skeleton'].float().cuda(gpu_id)
                # skeleton = sample['velocity'].float().cuda(gpu_id)
                imageData = sample['input_image'].float().cuda(gpu_id)
                t = skeleton.shape[2]
                y = sample['action'].data.item()
                label = torch.zeros(skeleton.shape[1], num_class)
                for i in range(0, skeleton.shape[1]):
                    input_clip = skeleton[:,i, :, :, :].reshape(1, t, -1)
                    imgClip = imageData[:, i, :, :, :]

                    # label_clip, _, _ = net(input_clip, t) # DY+BL+CL
                    label_clip, _, _ = net(input_clip,imgClip, t, fusion)
                    # label_clip, _, _ = net.dynamicsClassifier(input_clip, t)
                    label[i] = label_clip
                label = torch.mean(label, 0, keepdim=True)

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

            print('epoch:', epoch, 'Acc:%.4f'% Acc, 'count:',count, 'pred_cnt:', pred_cnt)
            ACC.append(Acc)

# data = {'LOSS':np.asarray(LOSS), 'Acc':np.asarray(ACC)}
# scipy.io.savemat('./matFile/Classifier_train_v12_test_v3_10cls.mat', mdict=data)

print('done')