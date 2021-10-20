from torch.utils.data import DataLoader
from utils import *
import scipy.io
from modelZoo.networks import *
from torch.optim import lr_scheduler
from scipy.spatial import distance
import torch.nn
from modelZoo.sparseCoding import *
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from dataset.NTU_viewProjection import *

gpu_id = 2
num_workers = 4
PRE = 0
dataset = 'NUCLA'
# dataset = 'NTU-D'

N = 40*2
Epoch = 150
T = 36
dataType = '2D'
clip = 'Multi'

P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()

# net = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)
# net.cuda(gpu_id)
net = DyanEncoder(Drr, Dtheta, lam=0.01, gpu_id=gpu_id).cuda(gpu_id)

modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
# saveModel = os.path.join(modelRoot, dataset, '/BinarizeSparseCode_m32A1')
saveModel = modelRoot + dataset + '/newDYAN/sparseCode_NUCLA_T36_fist001_openPose_multi/'
if not os.path.exists(saveModel):
    os.makedirs(saveModel)
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

elif dataset == 'NTU-D':
    nanList = list(np.load('./NTU_badList.npz')['x'])

    trainSet = NTURGBD_viewProjection(root_skeleton="/data/Yuexi/NTU-RGBD/skeletons/npy",
                                      root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType=dataType,
                                      clip=clip,
                                      phase='train', T=36, target_view='C002', project_view='C003', test_view='C001')

    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    valSet = NTURGBD_viewProjection(root_skeleton="/data/Yuexi/NTU-RGBD/skeletons/npy",
                                    root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType=dataType,
                                    clip=clip,
                                    phase='test', T=36, target_view='C002', project_view='C003', test_view='C001')
    valloader = DataLoader(valSet, batch_size=1, shuffle=True, num_workers=num_workers)


# optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-4, weight_decay=0.001, momentum=0.9)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, weight_decay=0.001, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 130], gamma=0.1)
criterion = torch.nn.MSELoss()

for epoch in range(0, Epoch+1):
    print('start training epoch:', epoch)
    lossVal = []
    loss1Val = []
    loss2Val = []
    for i, sample in enumerate(trainloader):
        optimizer.zero_grad()

        # skeleton_v1 = sample['target_skeleton']['normSkeleton'].float().cuda(gpu_id) # for single-clip
        # skeleton_v2 = sample['project_skeleton']['normSkeleton'].float().cuda(gpu_id)
        skeleton_v1 = sample['target_skeleton'].float().cuda(gpu_id)
        skeleton_v2 = sample['project_skeleton'].float().cuda(gpu_id)

        # print('shape:', skeleton_v1.shape, skeleton_v2.shape)
        # y = sample['cls'].data.item()
        # print('sample:', i, 'view1 info:', sample['target_info']['name_sample'], 'view2 info:', sample['project_info']['name_sample'])
        'Single Clip'
        """""
        t1 = skeleton_v1.shape[1]
        t2 = skeleton_v2.shape[1]

        input_v1 = skeleton_v1.reshape(1, t1, -1)
        input_v2 = skeleton_v2.reshape(1, t2, -1)

        coeff1, dict1 = net(input_v1, t1)
        output_v1 = torch.matmul(dict1, coeff1)
        coeff2, dict2 = net(input_v2, t2)
        output_v2 = torch.matmul(dict2, coeff2)

        loss1 = criterion(input_v1, output_v1)
        loss2 = criterion(input_v2, output_v2)
        """
        t1 = skeleton_v1.shape[2]
        t2 = skeleton_v2.shape[2]

        clipMSE1 = torch.zeros(skeleton_v1.shape[1]).cuda(gpu_id)
        clipMSE2 = torch.zeros(skeleton_v2.shape[1]).cuda(gpu_id)

        for clip in range(0, skeleton_v2.shape[1]):

            v1_clip = skeleton_v1[:,clip,:,:,:].reshape(1,t1,-1)
            v2_clip = skeleton_v2[:,clip,:,:,:].reshape(1,t2, -1)

            coeff1, dict1 = net(v1_clip, t1)
            output_v1 = torch.matmul(dict1, coeff1)
            coeff2, dict2 = net(v2_clip, t2)
            output_v2 = torch.matmul(dict2, coeff2)

            clipMSE1[clip] = criterion(output_v1, v1_clip)
            clipMSE2[clip] = criterion(output_v2, v2_clip)

        loss1 = torch.sum(clipMSE1)
        loss2 = torch.sum(clipMSE2)



        loss = loss1 + loss2
        # loss = loss1
        # loss2 = 0
        loss.backward()
        optimizer.step()

        lossVal.append(loss.data.item())
        loss1Val.append(loss1.data.item())
        loss2Val.append(loss2.data.item())

    loss_val = np.mean(np.array(lossVal))
    loss1_val = np.mean(np.array(loss1Val))
    loss2_val = np.mean(np.array(loss2Val))
    scheduler.step()

    print('Epoch:', epoch, '|total loss:', loss_val, '|view1', loss1_val, '|view2', loss2_val)
    if epoch % 10 == 0:
        torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, saveModel + str(epoch) + '.pth')

    if epoch%10 == 0:
        print('validating...')
        Error = []
        with torch.no_grad():
            for i, sample in enumerate(valloader):
                'single clip'
                """""
                skeleton = sample['input_skeleton']['normSkeleton'].float().cuda(gpu_id)
                t = skeleton.shape[1]
                input = skeleton.reshape(1, t, -1)

                coeff, dict = net(input,t)
                output = torch.matmul(dict, coeff)

                error = torch.norm(input - output)
                Error.append(error.data.item())
                """
                'multi-clip'
                inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
                t = inputSkeleton.shape[2]
                clipMSE = torch.zeros(inputSkeleton.shape[1]).cuda(gpu_id)
                for i in range(0, inputSkeleton.shape[1]):

                    input_clip = inputSkeleton[:,i, :, :, :].reshape(1, t, -1)
                    coeff, dict = net(input_clip, t)
                    output = torch.matmul(dict, coeff)

                    err = torch.norm(input_clip - output)
                    clipMSE[i] = err
                error = torch.sum(clipMSE)
                Error.append(error.data.item())

            print('epoch:', epoch, 'error:', np.mean(np.asarray(Error)))

print('done')
