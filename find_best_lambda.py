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
from modelZoo.sparseCoding import DyanEncoder
import time
from lossFunction import binaryLoss
# torch.backends.cudnn.enabled = False
from lossFunction import hashingLoss, CrossEntropyLoss
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

gpu_id = 0
num_workers = 2

dataset = 'NUCLA'
# dataset = 'NTU'
Alpha = 0.1
# for BI
lam1 = 2  # for CL
lam2 = 1 # for MSE
T = 36
N = 80*2
Epoch = 100
dataType = '2D'
# clip = 'Single'
clip = 'Multi'
fusion = False


modelRoot = '/data/Yuexi/Cross_new/'
num_class = 10
path_list = f"/data/N-UCLA_MA_3D/lists"
root_skeleton = '/data/N-UCLA_MA_3D/openpose_est'
# root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_3d'
trainSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='train', cam='2,1', T=T,
                               target_view='view_2', project_view='view_1', test_view='view_3')
# #
trainloader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=num_workers)

valSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T,
                              target_view='view_2',
                              project_view='view_1', test_view='view_3')

valloader = DataLoader(valSet, batch_size=1, shuffle=True, num_workers=num_workers)
# X = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
    # , 2.1, 2.2, 2.3, 2.4, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]
# X = [0.2, 0.3, 0.4, 0.5, 0.6]
X = [0.1, 1, 10, 100]
for num in range(0, len(X)):
    # x = X[num]
    x = 0.1
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    data1 = {'Drr': Drr.numpy(), 'Dtheta':Dtheta.numpy()}
    matName = '/data/Yuexi/Cross_view/find_lambda_' + str(x) + '_initalD_moreEP.mat'
    scipy.io.savemat(matName, mdict=data1)

    print('starting X=', X[num], 'lambda:', x)
    saveModel = modelRoot + '/find_lambda/lambda_moreEP_' + str(x) + '/'
    if not os.path.exists(saveModel):
        os.makedirs(saveModel)

    # net = classificationWSparseCode(num_class=10, Npole=1*N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2, fistaLam=x, gpu_id=gpu_id).cuda(gpu_id)
    # net = Fullclassification(num_class=num_class, Npole=1 * N + 1, num_binary=1 * N + 1, Drr=Drr, Dtheta=Dtheta, dim=2,
    #                          dataType=dataType, Inference=True,
    #                                                   gpu_id=gpu_id, fistaLam=x).cuda(gpu_id)
    net = DyanEncoder(Drr, Dtheta, x, gpu_id).cuda(gpu_id)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=0.001, momentum=0.9)
    # optimizer = torch.optim.SGD([{'params': filter(lambda x: x.requires_grad, net.sparseCoding.parameters()), 'lr': 1e-6},
    #      {'params': filter(lambda x: x.requires_grad, net.Classifier.parameters()), 'lr': 1e-3}], weight_decay=0.001,
    #     momentum=0.9)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
    Criterion = torch.nn.CrossEntropyLoss()
    mseLoss = torch.nn.MSELoss()
    start = time.time()
    for epoch in range(0, Epoch + 1):
        print('start training epoch:', epoch)
        lossVal = []
        lossCls = []
        lossBi = []
        lossMSE = []
        for i, sample in enumerate(trainloader):
            print('sample:', i)
            optimizer.zero_grad()
            input_v1 = sample['target_skeleton'].float().cuda(gpu_id)
            input_v2 = sample['project_skeleton'].float().cuda(gpu_id)

            y = sample['action'].cuda(gpu_id)
            t1 = input_v1.shape[2]
            t2 = input_v2.shape[2]

            label1 = torch.zeros(input_v1.shape[1], num_class).cuda(gpu_id)
            label2 = torch.zeros(input_v2.shape[1], num_class).cuda(gpu_id)
            # clipBI1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
            # clipBI2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)
            clipMSE1 = torch.zeros(input_v1.shape[1]).cuda(gpu_id)
            clipMSE2 = torch.zeros(input_v2.shape[1]).cuda(gpu_id)

            for clip in range(0, input_v2.shape[1]):
                v1_clip = input_v1[:, clip, :, :, :].reshape(1, t1, -1)
                v2_clip = input_v2[:, clip, :, :, :].reshape(1, t2, -1)
                # _, _, outClip_v1 = net.sparseCoding(v1_clip, t1)
                # _, _, outClip_v2 = net.sparseCoding(v2_clip, t2)
                _, _, outClip_v1 = net(v1_clip, t1)
                _, _, outClip_v2 = net(v2_clip, t2)
                clipMSE1[clip] = mseLoss(outClip_v1, v1_clip)
                clipMSE2[clip] = mseLoss(outClip_v2, v2_clip)

                # label1[clip] = label_clip1
                # label2[clip] = label_clip2

            # label1 = torch.mean(label1, 0, keepdim=True)
            # label2 = torch.mean(label2, 0, keepdim=True)

            # loss1 = lam1 * Criterion(label1, y) + lam2 * (torch.mean(clipMSE1))
            # loss2 = lam1 * Criterion(label2, y) + lam2 * (torch.mean(clipMSE2))
            loss1 = torch.mean(clipMSE1)
            loss2 = torch.mean(clipMSE2)
            loss = loss1 + loss2
            loss.backward()

            optimizer.step()
            lossVal.append(loss.data.item())
            # lossCls.append(Criterion(label1, y).data.item() + Criterion(label2, y).data.item())
            lossMSE.append((torch.mean(clipMSE1)).data.item() + (torch.mean(clipMSE2)).data.item())
        end = time.time()
        loss_val = np.mean(np.array(lossVal))
        # print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|mse:', np.mean(np.array(lossMSE)))
        print('epoch:', epoch, '|loss:', loss_val, 'running time (hr):', (end-start)/3600)
        scheduler.step()
        if epoch % 10 == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')
        # if epoch % 5 == 0:
        if epoch == 25:
            print('start validating:')
            count = 0
            pred_cnt = 0
            Acc = []
            Error = []
            with torch.no_grad():
                for i, sample in enumerate(valloader):
                    # input = sample['test_view_multiClips'].float().cuda(gpu_id)
                    inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)

                    t = inputSkeleton.shape[2]
                    y = sample['action'].data.item()
                    label = torch.zeros(inputSkeleton.shape[1], num_class)
                    clipMSE = torch.zeros(inputSkeleton.shape[1]).cuda(gpu_id)
                    Coeff = []
                    for ii in range(0, inputSkeleton.shape[1]):
                        input_clip = inputSkeleton[:, ii, :, :, :].reshape(1, t, -1)
                        c, d, output_clip = net(input_clip, t)
                        Coeff.append(c.cpu().numpy())
                        # label[ii] = label_clip
                        err = torch.norm(input_clip - output_clip)
                        clipMSE[ii] = err
                    Coeff = np.asarray(Coeff)
                    # rr = net.sparseCoding.rr.cpu().numpy()
                    # theta = net.sparseCoding.theta.cpu().numpy()
                    rr = net.rr.cpu().numpy()
                    theta = net.theta.cpu().numpy()
                    dictionary = d.cpu().numpy()

                    if i == 5:
                        data = {'Coeff':Coeff, 'rr':rr, 'theta':theta, 'dictionary':dictionary}
                        name = 'lambda_NEW_moreEP_' + str(x) + '.mat'
                        scipy.io.savemat('/data/Yuexi/Cross_view/'+ name, mdict=data)

                    error = torch.sum(clipMSE)
                Error.append(error.data.item())
                    # label = torch.mean(label, 0, keepdim=True)
                    # pred = torch.argmax(label).data.item()
                    #
                    # count += 1
                    #
                    # if pred == y:
                    #     pred_cnt += 1

                # Acc = pred_cnt / count

                # print('epoch:', epoch, 'Acc:%.4f' % Acc, 'count:', count, 'pred_cnt:', pred_cnt)
                print('epoch:', epoch, 'error:', np.mean(np.asarray(Error)))
    print('done experiment:', num)

print('done all')


