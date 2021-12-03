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
# torch.backends.cudnn.enabled = False
from lossFunction import hashingLoss, CrossEntropyLoss
import time
import stats

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
fistaLam = 0.3
gpu_id = 1
num_workers = 4
PRE = 0
print('gpu_id: ',gpu_id)
print('num_workers: ',num_workers)
print('fistaLam: ',fistaLam)
T = 36
print('T: ',T)
dataset = 'NUCLA'
# dataset = 'NTU'
Alpha = 0.1
# for BI
lam1 = 2  # for CL
lam2 = 1 # for MSE

N = 40*2
Epoch = 100
dataType = '3D'
dim = 3

# clip = 'Single'
clip = 'Multi'
fusion = False

P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

modelRoot = '/home/balaji/Documents/code/RSL/CS_CV/Cross-View/models/'
# saveModel = os.path.join(modelRoot, dataset, '/BinarizeSparseCode_m32A1')
# saveModel = modelRoot + dataset + '/2Stream/train_t36_CV_openpose_testV3_lam1051/'
saveModel = modelRoot + dataset + '/1201/CV_dynamicsStream_fista05_reWeighted_sqrC_T72/'
fig_save_path = os.path.join(saveModel, 'plots')
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

lst_path1 = fig_save_path + '/sparse_data_c1.lst'
lst_writer1 = open(lst_path1, mode='w+')
lst_path2 = fig_save_path + '/sparse_data_c2.lst'
lst_writer2 = open(lst_path2, mode='w+')

lst_pathb1 = fig_save_path + '/sparse_data_b1.lst'
lst_writerb1 = open(lst_pathb1, mode='w+')
lst_pathb2 = fig_save_path + '/sparse_data_b2.lst'
lst_writerb2 = open(lst_pathb2, mode='w+')

lst_path = fig_save_path + '/sparse_data_c.lst'
lst_writer = open(lst_path, mode='w+')
lst_pathb = fig_save_path + '/sparse_data_b.lst'
lst_writerb = open(lst_pathb, mode='w+')

map_location = torch.device(gpu_id)

# modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'

'load pre-trained DYAN'
# preTrained = modelRoot + dataset + '/newDYAN/BinarizeSparseCode_hardGumbel_v2/'
#
# stateDict = torch.load(os.path.join(preTrained, '10.pth'), map_location=map_location)['state_dict']
# Drr = stateDict['sparseCoding.rr']
# Dtheta = stateDict['sparseCoding.theta']
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
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_3d'
    trainSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='train', cam='2,1', T=T,
                                   target_view='view_2', project_view='view_1', test_view='view_3')
    # #
    trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

    valSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T,
                                  target_view='view_2',
                                  project_view='view_1', test_view='view_3')
    valloader = DataLoader(valSet, batch_size=1, shuffle=True, num_workers=num_workers)

elif dataset == 'NTU':
    num_class = 120
    nanList = list(np.load('./NTU_badList.npz')['x'])
    if dataType == '3D':
        root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    else:
        root_skeleton = "/data/NTU-RGBD/poses"

    trainSet = NTURGBD_viewProjection(root_skeleton=root_skeleton,
                                root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
                                phase='train', T=36, target_view='C002', project_view='C003', test_view='C001')

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    valSet = NTURGBD_viewProjection(root_skeleton=root_skeleton,
                                root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
                                phase='test', T=36, target_view='C002', project_view='C003', test_view='C001')
    valloader = DataLoader(valSet, batch_size=1, shuffle=True, num_workers=num_workers)


# net = classificationHead(num_class=10, Npole=(N+1)).cuda(gpu_id)
# net = classificationWBinarization(num_class=10, Npole=(2*N+1), num_binary=(N+1)).cuda(gpu_id)
# net = classificationWSparseCode(num_class=10, Npole=2*N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2,gpu_id=gpu_id).cuda(gpu_id)
net = Fullclassification(num_class=num_class, Npole=2*N+1, num_binary=2*N+1, Drr=Drr, Dtheta=Dtheta, dim=dim, dataType=dataType, Inference=True,
                         gpu_id=gpu_id, fistaLam=fistaLam).cuda(gpu_id)
# kinetics_pretrain = './pretrained/i3d_kinetics.pth'
# net = twoStreamClassification(num_class=num_class, Npole=(2*N+1), num_binary=(N+1), Drr=Drr, Dtheta=Dtheta,
#                                   PRE=0, dim=2, gpu_id=gpu_id, dataType=dataType, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

# net = RGBAction(num_class=num_class, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

net.train()
# for m in net.sparseCoding.parameters():
#     m.requires_grad = False

# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-4, weight_decay=0.001, momentum=0.9)
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, weight_decay=0.001, momentum=0.9)


# optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.dynamicsClassifier.parameters()), 'lr':1e-3},
# {'params':filter(lambda x: x.requires_grad, net.RGBClassifier.parameters()), 'lr':1e-4}], weight_decay=0.001, momentum=0.9)

optimizer = torch.optim.SGD([{'params':filter(lambda x: x.requires_grad, net.sparseCoding.parameters()), 'lr':1e-6},
                             {'params':filter(lambda x: x.requires_grad, net.Classifier.parameters()), 'lr':1e-3}], weight_decay=0.001, momentum=0.9)


scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()
# Criterion = torch.nn.BCELoss()
LOSS = []
ACC = []
print('training dataset:', dataset)
print('alpha:', Alpha, 'lam1:', lam1, 'lam2:', lam2)
# print('cls:bi:reconst=2:0.3:1')
torch.autograd.set_detect_anomaly(True)
for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    start = time.time()
    lossVal = []
    lossCls = []
    lossBi = []
    lossMSE = []
    for i, sample in enumerate(trainloader):

        # print('sample:', i)
        optimizer.zero_grad()
        # input_v1 = sample['target_view_multiClips'].float().cuda(gpu_id)
        # input_v2 = sample['project_view_multiClips'].float().cuda(gpu_id)
        '2S'

        input_v1 = sample['target_skeleton'].float().cuda(gpu_id)
        input_v2 = sample['project_skeleton'].float().cuda(gpu_id)
        input_v1_img = sample['target_image'].float().cuda(gpu_id)
        input_v2_img = sample['project_image'].float().cuda(gpu_id)

        # input_v1 = sample['target_velocity'].float().cuda(gpu_id)
        # input_v2 = sample['project_velocity'].float().cuda(gpu_id)


        y = sample['action'].cuda(gpu_id)
        t1 = input_v1.shape[2]
        t2 = input_v2.shape[2]

        label1 = torch.zeros(input_v1.shape[1], num_class).cuda(gpu_id)
        label2 = torch.zeros(input_v2.shape[1], num_class).cuda(gpu_id)
        clipBI1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
        clipBI2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)
        clipMSE1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
        clipMSE2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)

        label_rgb1 = torch.zeros(input_v1.shape[1], num_class).cuda(gpu_id)
        label_rgb2 = torch.zeros(input_v2.shape[1], num_class).cuda(gpu_id)
        label_dym1 = torch.zeros(input_v1.shape[1], num_class).cuda(gpu_id)
        label_dym2 = torch.zeros(input_v2.shape[1], num_class).cuda(gpu_id)

        for clip in range(0, input_v2.shape[1]):

            v1_clip = input_v1[:,clip,:,:,:].reshape(1,t1,-1)
            v2_clip = input_v2[:,clip,:,:,:].reshape(1,t2, -1)

            'two stream model'

            # img1_clip = input_v1_img[:,clip,:,:,:]
            # img2_clip = input_v2_img[:,clip,:,:,:]
            # label_clip1, b1, outClip_v1 = net(v1_clip, img1_clip, t1, fusion)
            # label_clip2, b2, outClip_v2 = net(v2_clip, img2_clip, t2, fusion)
            # label_clip1 = net(img1_clip)
            # label_clip2 = net(img2_clip)

            'Full Model'
            label_clip1, b1, outClip_v1, c1 = net(v1_clip, t1)
            label_clip2, b2, outClip_v2, c2 = net(v2_clip, t2)

            # if i%20==0 and clip==0:
            #     stats.write_data(lst_writer1, c1, delimiter=',')
            #     stats.write_data(lst_writer2, c2, delimiter=',')
            #     stats.write_data(lst_writerb1, b1, delimiter=',')
            #     stats.write_data(lst_writerb2, b2, delimiter=',')

            bi_gt1 = torch.zeros_like(b1).cuda(gpu_id)
            bi_gt2 = torch.zeros_like(b2).cuda(gpu_id)

            if fusion:
                label_rgb1[clip] = label_clip1['RGB']
                label_rgb2[clip] = label_clip2['RGB']
                label_dym1[clip] = label_clip1['Dynamcis']
                label_dym2[clip] = label_clip2['Dynamcis']

            else:
                label1[clip] = label_clip1
                label2[clip] = label_clip2


            # clipBI1[clip] = torch.sum(b1)/((2*N+1)*50)
            # clipBI1[clip] = binaryLoss(b1, gpu_id)
            clipBI1[clip] = L1loss(b1, bi_gt1)
            clipMSE1[clip] = mseLoss(outClip_v1, v1_clip)

            # clipBI2[clip] = binaryLoss(b2, gpu_id)
            # clipBI2[clip] = torch.sum(b2)/((2*N+1)*50)
            # clipBI1[clip] = torch.norm(b2)

            clipBI2[clip] = L1loss(b2, bi_gt2)
            clipMSE2[clip] = mseLoss(outClip_v2, v2_clip)

            """""
            'DYAN+CL'

            label_clip1, outClip_v1 = net(v1_clip, t1)
            label_clip2, outClip_v2 = net(v2_clip, t2)
            clipMSE1[clip] = mseLoss(outClip_v1, v1_clip)
            clipMSE2[clip] = mseLoss(outClip_v2, v2_clip)

            label1[clip] = label_clip1
            label2[clip] = label_clip2
            """""
        if fusion:
            label_rgb1 = torch.mean(label_rgb1, 0, keepdim=True)
            label_rgb2 = torch.mean(label_rgb2, 0, keepdim=True)
            label_dym1 = torch.mean(label_dym1, 0, keepdim=True)
            label_dym2 = torch.mean(label_dym2, 0, keepdim=True)

            y_rgb1 = torch.tensor(torch.argmax(label_rgb1).data.item()).cuda(gpu_id).unsqueeze(0)
            y_rgb2 = torch.tensor(torch.argmax(label_rgb2).data.item()).cuda(gpu_id).unsqueeze(0)

            LossCl1 = Criterion(label_rgb1, y) + Criterion(label_dym1, y) + Criterion(label_dym1, y_rgb1)
            LossCl2 = Criterion(label_rgb2, y) + Criterion(label_dym2, y) + Criterion(label_dym2, y_rgb2)
            loss1 = lam1 * LossCl1 + Alpha * (torch.mean(clipBI1)) + lam2 * (torch.mean(clipMSE1))
            loss2 = lam1 * LossCl2 + Alpha * (torch.mean(clipBI2)) + lam2 * (torch.mean(clipMSE2))

            label1 = 0.5*label_rgb1 + 0.5*label_dym1
            label2 = 0.5*label_rgb2 + 0.5*label_dym2


        else:
            label1 = torch.mean(label1, 0, keepdim=True)
            label2 = torch.mean(label2, 0, keepdim=True)

        loss1 = lam1 * Criterion(label1, y) + Alpha * (torch.mean(clipBI1)) + lam2 * (torch.mean(clipMSE1))
        loss2 = lam1 * Criterion(label2, y) + Alpha * (torch.mean(clipBI2)) + lam2 * (torch.mean(clipMSE2))
        # print('bi loss:', torch.sum(clipBI1))
        # loss1 = Criterion(label1, y)
        # loss2 = Criterion(label2, y)
        # loss1 = lam1 * Criterion(label1, y) + lam2 * (torch.sum(clipMSE1))
        # loss2 = lam1 * Criterion(label2, y) + lam2 * (torch.sum(clipMSE2))
        loss = loss1 + loss2
        loss.backward()
        # pdb.set_trace()
        # print('bin fc wt grad, ', net.BinaryCoding.gate_network[0].weight.grad)
        # print('rr.grad:', net.sparseCoding.rr.grad, 'theta.grad:', net.sparseCoding.theta.grad)
        # print('cls.grad:', net.Classifier.FC.weight.grad)
        optimizer.step()
        lossVal.append(loss.data.item())
        lossCls.append(Criterion(label1, y).data.item() + Criterion(label2, y).data.item())
        lossBi.append((torch.mean(clipBI1)).data.item() + (torch.mean(clipBI2)).data.item())
        lossMSE.append((torch.mean(clipMSE1)).data.item() + (torch.mean(clipMSE2)).data.item())

    loss_val = np.mean(np.array(lossVal))
    # LOSS.append(loss_val)
    # print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|mse:', np.mean(np.array(lossMSE)))
    # print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|Bi:', np.mean(np.array(lossBi)))
    train_mins = (time.time() - start) / 60 # 1 epoch training 
    print(' epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|Bi:', np.mean(np.array(lossBi)),
          '|mse:', np.mean(np.array(lossMSE)))
    print('time: ', round(train_mins, 3)) 
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
            for j, sample in enumerate(valloader):
                # input = sample['test_view_multiClips'].float().cuda(gpu_id)
                inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)

                # inputSkeleton = sample['test_velocity'].float().cuda(gpu_id)
                inputImage = sample['input_image'].float().cuda(gpu_id)

                t = inputSkeleton.shape[2]
                y = sample['action'].data.item()
                label = torch.zeros(inputSkeleton.shape[1], num_class)
                
                for i in range(0, inputSkeleton.shape[1]):

                    input_clip = inputSkeleton[:,i, :, :, :].reshape(1, t, -1)
                    inputImg_clip = inputImage[:,i, :, :, :]
                    # label_clip, _, _ = net(input_clip, t) # DY+BL+CL
                    # label_clip, _ = net(input_clip, t) # DY+CL

                    if fusion:
                        label_clip, _, _ = net.dynamicsClassifier(input_clip, t) # two stream, dynamcis branch
                    else:
                        # label_clip, _, _ = net(input_clip, inputImg_clip, t, fusion)
                        # label_clip,_ = net(input_clip, t) #DY
                        label_clip, b, out_clip, c = net(input_clip, t) # DY+BI

                        # if j%10==0 and i==0:
                        #     stats.write_data(lst_writer, c, delimiter=',')
                        #     stats.write_data(lst_writerb, b, delimiter=',')

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