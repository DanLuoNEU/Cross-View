from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import *
import scipy.io
from modelZoo.networks import *
from torch.autograd import Variable
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
import time
from lossFunction import binaryLoss
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

gpu_id = 6
num_workers = 1
PRE = 0

T = 36
dataset = 'NUCLA'
# dataset = 'NTU'
Alpha = 0.1
lam1 = 1
lam2 = 1
<<<<<<< HEAD
N = 80*2
# N = 42*2 # setup1
# N = 53*2    #setup2
# N = 50*2    # setup3
# Epoch = 150
=======
N = 40*2
Epoch = 150
>>>>>>> b6ead8a973a18436106ea495f2d5e245e8a794ab
dataType = '2D'
# clip = 'Single'
clip = 'Multi'
fusion = False
# fusion = True
<<<<<<< HEAD
#
modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'

modelPath = modelRoot + dataset + '/1110/dynamicStream_fista01_reWeighted_noBI_sqrC_T36_UCLA/'
# modelPath = modelRoot + dataset + '/1102/CV_2Stream_fista01_reWeighted_noBI_sqrC_T36_UCLA/'
# modelPath = modelRoot + dataset + '/1029/CV_2Stream_fista01_reWeighted_sqrC_T36_NTU/'
# modelPath = modelRoot + dataset + '/1129/CV_fista02_reducedD_v1/'
# modelPath = modelRoot + dataset + '/3D/'
map_location = torch.device(gpu_id)
stateDict = torch.load(os.path.join(modelPath, '100.pth'), map_location=map_location)['state_dict']
# Drr = stateDict['dynamicsClassifier.sparseCoding.rr']
# Dtheta = stateDict['dynamicsClassifier.sparseCoding.theta']

Drr = stateDict['sparseCoding.rr']
Dtheta = stateDict['sparseCoding.theta']
# dictionary = scipy.io.loadmat('/home/dan/ws/2021-CrossView/matfiles/1115_coeff-wi-bin/Dic_clean_20211119.mat')
# Drr = torch.tensor(dictionary['r_clean']).squeeze(0)
# Dtheat = torch.tensor(dictionary['theta_clean']).squeeze(0)


data = {'Drr': Drr.cpu().numpy(), 'Dtheta':Dtheta.cpu().numpy()}
# scipy.io.savemat('/data/Yuexi/Cross_view/1129/setup1/UCLA_fista02_reduce_dictionary_v1.mat', mdict=data)
# Drr = stateDict['sparseCoding.rr']
# Dtheta = stateDict['sparseCoding.theta']
=======

modelRoot = './ModelFile/crossViewModel/'
# modelPath = modelRoot + dataset + '/CS/2S_Multi_500_openpose_lam1051/'
# modelPath = modelRoot + dataset + '/2Stream/train_t36_CV_openpose_testV3_lam1051/'
# modelPath = modelRoot + dataset + '/2Stream/multiClip_lam1051_testV2_CV/'
# modelPath = modelRoot + dataset + '/DYOnly_Multi_CV/'
modelPath = modelRoot + dataset + '/newDYAN/CV_dynamicsStream_CLOnly/'
map_location = torch.device(gpu_id)
stateDict = torch.load(os.path.join(modelPath, '100.pth'), map_location=map_location)['state_dict']
# Drr = stateDict['dynamicsClassifier.sparsecoding.l1.rr']
# Dtheta = stateDict['dynamicsClassifier.sparsecoding.l1.theta']

Drr = stateDict['sparseCoding.rr']
Dtheta = stateDict['sparseCoding.theta']


>>>>>>> b6ead8a973a18436106ea495f2d5e245e8a794ab


if dataset == 'NUCLA':
    num_class = 10
<<<<<<< HEAD
    path_list = f"/data/N-UCLA_MA_3D/lists"
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    'CS:'
    # print('test dataset:', dataset, 'cross subject experiment')
=======
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    'CS:'
>>>>>>> b6ead8a973a18436106ea495f2d5e245e8a794ab
    # testSet = NUCLAsubject(root_list=path_list, dataType=dataType, clip=clip, phase='test', T=T)
    # testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=num_workers)


    'CV:'
<<<<<<< HEAD
    print('test dataset:', dataset, 'cross view experiment')
    testSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T,
=======
    testSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='1,2', T=T,
>>>>>>> b6ead8a973a18436106ea495f2d5e245e8a794ab
                                  target_view='view_2',
                                  project_view='view_1', test_view='view_3')
    testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=num_workers)

elif dataset == 'NTU':
    num_class = 60
    # root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    if dataType == '3D':
        root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    else:
        root_skeleton = "/data/NTU-RGBD/poses"
    nanList = list(np.load('./NTU_badList.npz')['x'])
    'CS:'

<<<<<<< HEAD
    testSet = NTURGBDsubject(root_skeleton, nanList, dataType=dataType, clip=clip, phase='test', T=36)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    'CV:'
    # testSet = NTURGBD_viewProjection(root_skeleton, root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
    #                             phase='test', T=36, target_view='C002', project_view='C001', test_view='C003')
    # testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=num_workers)

# net = classificationWSparseCode(num_class=10, Npole=N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2, fistaLam=0.1, gpu_id=gpu_id).cuda(gpu_id)
net = Fullclassification(num_class=num_class, Npole=1*N+1, num_binary=1*N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True,
                         gpu_id=gpu_id, fistaLam=0.1).cuda(gpu_id)

# kinetics_pretrain = './pretrained/i3d_kinetics.pth'
# net = twoStreamClassification(num_class=num_class, Npole=(1*N+1), num_binary=(N+1), Drr=Drr, Dtheta=Dtheta,
#                             dim=2, gpu_id=gpu_id, inference=True, fistaLam=0.1, dataType=dataType, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

# net = RGBAction(num_class=num_class, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
net.load_state_dict(stateDict)
# net.eval()
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

count = 0
pred_cnt = 0
ACC = []
classLabel = [[] for i in range(0, num_class)]
classGT = [[] for i in range(0, num_class)]
# Coeff_v3_action01 = []
# Coeff_v1_action01 = []
# Coeff_v2_action01 = []
# Coeff_v3_action05 = []
# binaryCode = [[] for i in range(0, num_class)]
# origCoeff = [[] for i in range(0, num_class)]
binaryCode = []
origCoeff = []

# origY = []
# origImage = []
with torch.no_grad():
    for s, sample in enumerate(testloader):
        # print('testing:', s)
        # input = sample['test_view_multiClips'].float().cuda(gpu_id)
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        # inputSkeleton = sample['project_skeleton'].float().cuda(gpu_id)

        # inputSkeleton = sample['test_velocity'].float().cuda(gpu_id)
        inputImage = sample['input_image'].float().cuda(gpu_id)
=======
    # testSet = NTURGBDsubject(root_skeleton, nanList, dataType=dataType, clip=clip, phase='test', T=36)
    # testloader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    'CV:'
    testSet = NTURGBD_viewProjection(root_skeleton, root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
                                phase='test', T=36, target_view='C002', project_view='C001', test_view='C003')
    testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=num_workers)

# net = classificationWSparseCode(num_class=10, Npole=2*N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2,gpu_id=gpu_id).cuda(gpu_id)
net = Fullclassification(num_class=num_class, Npole=2*N+1, num_binary=2*N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True, gpu_id=gpu_id).cuda(gpu_id)
# kinetics_pretrain = './pretrained/i3d_kinetics.pth'
# net = twoStreamClassification(num_class=num_class, Npole=(N+1), num_binary=(N+1), Drr=Drr, Dtheta=Dtheta,
#                                   PRE=0, dim=2, gpu_id=gpu_id, dataType=dataType, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

# net = RGBAction(num_class=num_class, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
net.load_state_dict(stateDict)
net.eval()


count = 0
pred_cnt = 0
Acc = []
classLabel = [[] for i in range(0, num_class)]
classGT = [[] for i in range(0, num_class)]
with torch.no_grad():

    for s, sample in enumerate(testloader):
        # input = sample['test_view_multiClips'].float().cuda(gpu_id)
        'Multi'
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        # inputSkeleton = sample['project_skeleton'].float().cuda(gpu_id)
        # inputImage = sample['input_image'].float().cuda(gpu_id)

        'Single clip'
        # inputSkeleton = sample['input_skeleton']['normSkeleton'].unsqueeze(0).float().cuda(gpu_id)
        # inputImage = sample['input_image'].unsqueeze(0).float().cuda(gpu_id)
>>>>>>> b6ead8a973a18436106ea495f2d5e245e8a794ab

        t = inputSkeleton.shape[2]
        y = sample['action'].data.item()
        label = torch.zeros(inputSkeleton.shape[1], num_class)
<<<<<<< HEAD

        # video_name = sample['sample_name'][0]

        # if sample['sample_name'][0] =='a02_s10_e02':
        #     print('check')
        result_bi = []
        result_coffe = []
        result_y = []

        start = time.time()
        for i in range(0, inputSkeleton.shape[1]):
        # for i in range(0, 1):

            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            inputImg_clip = inputImage[:, i, :, :, :]
            # label_clip, _, _ = net(input_clip, t) # DY+BL+CL
            # label_clip, _ = net(input_clip, t) # DY+CL

            if fusion:
                label_clip, _, _ = net.dynamicsClassifier(input_clip, t)  # two stream, dynamcis branch
            else:
                # label_clip, _, _,coeff, d = net(input_clip, inputImg_clip, t, fusion)
                # label_clip, _ = net(input_clip, t) #DY
                label_clip, bi, yr, coeff, d = net(input_clip, t)# DY+BI
                # binaryCode = bi.cpu().numpy()
                # mdata = {'binaryCode':binaryCode}
                # scipy.io.savemat('/data/Yuexi/Cross_view/1114/gumbel_binarycode.mat', mdict=mdata)
                # print('check')

                # if y == 4:
                #     Coeff_v2_action01.append(coeff.cpu().numpy())
                    # Coeff_v2_action01.append(coeff.cpu().numpy())

                if y == 5:
                #     Coeff_v3_action02.append(coeff.cpu().numpy())
                    result_bi.append(bi.cpu().numpy())
                    result_coffe.append(coeff.cpu().numpy())
                    # result_y.append(inputImg_clip.cpu().numpy())

            label[i] = label_clip
        label = torch.mean(label, 0, keepdim=True)
        end = time.time()
        print('time:', (end-start), 'time/clip:', (end-start)/inputSkeleton.shape[1])
        # c, _ = Encoder.forward2(input, T)
        # c = c.reshape(1, N + 1, int(input.shape[-1]/2), 2)
        # label = net(c)  # CL only
        # label, _ = net(c) # 'BI + CL'

        # label, _ = net(input, T) # 'DY + CL'

        pred = torch.argmax(label).data.item()
        print('sample:',s, 'pred:', pred, 'gt:', y)
        count += 1
        # if pred1 == y:
        #     pred_cnt +=1
        # elif pred2 == y:
        #     pred_cnt += 1

        if pred == y:
            pred_cnt += 1
            binaryCode = binaryCode+result_bi
            origCoeff = origCoeff + result_coffe
            # origY = origY + result_y

        if len(binaryCode) ==12:
            origCoeff = np.asarray(origCoeff)
            binaryCode = np.asarray(binaryCode)
            # origY = np.asarray(origY)
            # origY0_6 = np.asarray(origY[0:6])
            # origY7_12 = np.asarray(origY[6:12])
            # origY12_18 = np.asarray(origY[12:])
            # dict = {'bi_action': binaryCode, 'coeff_action': origCoeff, 'origY0_6': origY0_6,
            #         'origY7_12': origY7_12, 'origY12_8':origY12_18}
            dict = {'bi_action': binaryCode, 'coeff_action':origCoeff}
            
            scipy.io.savemat('/data/Yuexi/Cross_view/1129/setup1/coeff_bi_v3_action05.mat', mdict=dict)

            print('check')

    #     binaryCode[y].append(result_bi)
    #     origCoeff[y].append(result_coffe)
    #
    # dict = {'bi_action':binaryCode, 'coeff_action':origCoeff}
    #
    # scipy.io.savemat('/data/Yuexi/Cross_view/1129/coeff_bi_v1_all.mat', mdict=dict)






        # for n in range(0, label.shape[0]):
        #     pred = torch.argmax(label[n]).data.item()
        #     if pred == y[0]:
        #         count+= 1
        # acc = count/label.shape[0]
        # Acc.append(acc)
        # Acc = count/valSet.__len__()
    Acc = pred_cnt / count

    # print('Acc:%.4f' % Acc, 'count:', count, 'pred_cnt:', pred_cnt)
    # ACC.append(Acc)
=======
        """""
        if y == 0 and sample['sample_name'][0] == 'a01_s07_e00':
        # for i in range(0, inputSkeleton.shape[1]):
            for i in range(2, 3):


                input_clip = inputSkeleton[:,i, :, :, :].reshape(1, t, -1)
                input_clip = input_clip[:,:,1].unsqueeze(2)

                coeff, dict = net.dynamicsClassifier.sparsecoding.l1(input_clip, t)
                # weightPoles(coeff, Drr, Dtheta, dict)

                data = {'dict':dict.cpu().numpy(), 'coeff':coeff.cpu().numpy(),
                        'r':Drr.cpu().numpy(), 'theta':Dtheta.cpu().numpy()}
                scipy.io.savemat('./plotDict/model3_TestV2/view1Data_train_y0_dim1.mat', mdict=data)
                print('check')
        """""

        for i in range(0, inputSkeleton.shape[1]):
            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            # input_clip = input_clip[:, :, 1].unsqueeze(2)
            # inputImg_clip = inputImage[:,i, :, :, :]
            label_clip, _, _ = net(input_clip, t) # DY+BI+CL
            # label_clip, _ = net(input_clip, t) # DY+CL
            
            # if fusion:
            #     label_clip, _, _ = net.dynamicsClassifier(input_clip, t) # two stream, dynamcis branch
            #     # label_clip = net.RGBClassifier(inputImg_clip)
            # else:
            #     # label_clip, _, _ = net(input_clip, inputImg_clip, t, fusion)
            #     label_clip, _ = net(input_clip, t)  # DY + CL
            #     # label_clip = net(inputImg_clip)

            label[i] = label_clip
        label = torch.mean(label, 0, keepdim=True)

        pred = torch.argmax(label).data.item()

        count += 1
        classGT[y].append(y)
        if pred == y:
            classLabel[y].append(pred)
            pred_cnt += 1
        print('sample:',s, 'gt:', y, 'pred:', pred)
    Acc = pred_cnt/count

print('Acc:', Acc, 'total sample:', count, 'correct preds:', pred_cnt)
>>>>>>> b6ead8a973a18436106ea495f2d5e245e8a794ab

print('done')
