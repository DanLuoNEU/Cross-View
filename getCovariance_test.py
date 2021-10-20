from modelZoo.networks import keyframeProposalNet
from torch.utils.data import DataLoader, Dataset
from test_jhmdb import test_val
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from modelZoo.DyanOF import creatRealDictionary, OFModel
import time
from JHMDB_dloader import *
from utils import *
import torch.nn as nn
import random
torch.manual_seed(1)
random.seed(1)

T = 40
FRA = T
gpu_id = 1
N = 40*4

dataRoot = '/data/Yuexi/JHMDB'
trainAnnot, testAnnot,_ = get_train_test_annotation(dataRoot)
valSet = jhmdbDataset(trainAnnot, testAnnot, T=T, split='val', if_occ=False)
valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=8)

"""""
'loading poseFeat model'
poseFeatModel = '/home/yuexi/Documents/poseModel/poseDYAN/reconstructionOnly/2DJhmdb_T40_lam001/PoseDyan_xyz_20.pth'
poseFeat_stateDict = torch.load(poseFeatModel)
stateDict = poseFeat_stateDict['state_dict']
Dtheta = stateDict['l1.theta'].cuda(gpu_id)
Drr = stateDict['l1.rr'].cuda(gpu_id)
model = OFModel(Drr, Dtheta, T, gpu_id)
model.load_state_dict(stateDict)
model.cuda(gpu_id)
"""""


'loading imgFeat model'
imgFeatModel = '/home/yuexi/Documents/keyFrameModel/RealData/JHMDB/imgFeatDYAN/'
imgFeat_stateDict = torch.load(os.path.join(imgFeatModel, '50.pth'))['state_dict']
Drr = imgFeat_stateDict['l1.rr']
Dtheta = imgFeat_stateDict['l1.theta']
model = OFModel(Drr, Dtheta, T, gpu_id)
model.load_state_dict(imgFeat_stateDict)
model.cuda(gpu_id)

Dictionary = creatRealDictionary(T, Drr, Dtheta, gpu_id)
t_init = time.time()
ddt = torch.matmul(Dictionary, Dictionary.t())
ddt_inv = torch.inverse(ddt)

a = torch.matmul(Dictionary.t(), ddt_inv)
t_init_end = time.time() - t_init

'loading k-fpn'
modelFolder = '/home/yuexi/Documents/keyFrameModel/RealData/JHMDB/resnet18'
modelFile = os.path.join(modelFolder, 'lam41_83.pth')

state_dict = torch.load(modelFile)['state_dict']

DrrImg = state_dict['Drr']
DthetaImg = state_dict['Dtheta']
K_FPN = keyframeProposalNet(numFrame=T, Drr=DrrImg, Dtheta=DthetaImg, gpu_id=gpu_id, backbone='Resnet18')
newDict = K_FPN.state_dict()
pre_dict = {k: v for k, v in state_dict.items() if k in newDict}
newDict.update(pre_dict)
K_FPN.load_state_dict(newDict)
for param in K_FPN.parameters():
    param.requires_grad = False
K_FPN.cuda(gpu_id)

Time = []
Covariance = []
C_FISTA = []
Cov_Coeff = []
Coeff_LS = []
Error = []
meanCov = scipy.io.loadmat('./matfile/CovarianceMatrix_fullSequence.mat')['meanCov']
meanCov = torch.Tensor(meanCov).cuda(gpu_id)
lam = 0.001
t_init2 = time.time()
# dtd = torch.matmul(Dictionary.t(), Dictionary)
# inv = torch.matmul(meanCov, torch.inverse(torch.matmul(dtd, meanCov) + lam*I))

dtd = torch.matmul(torch.matmul(Dictionary,meanCov), Dictionary.t())
I = torch.eye(dtd.shape[1]).cuda(gpu_id)
inv = torch.inverse(dtd + lam*I)
b = torch.matmul(meanCov, torch.matmul(Dictionary.t(), inv))
t_init2_end = time.time() - t_init2

print('LS inverse time:', t_init_end, 'COV inverse time:', t_init2_end)
with torch.no_grad():
    for i, sample in enumerate(valloader):
        print('test sample:', i)
        img_data = sample['imgSequence_to_use']
        sequence_to_use = sample['sequence_to_use']
        inputData = img_data[0].cuda(gpu_id)
        if len(inputData.shape) == 5:
            inputData = inputData.squeeze(0)
        else:
            inputData = inputData

        imgFeature, _, _ = K_FPN.forward(inputData)
        """""
        out = K_FPN.forward2(imgFeature, 3*1)
        s = out[0, :]
        
        key_ind = (s>=0.99).nonzero().squeeze(1)
        key_list = list(key_ind.cpu().numpy())
        """
        #
        inputFeat = imgFeature.view(T, -1).unsqueeze(0)
        # inputFeat = sequence_to_use.reshape(sequence_to_use.shape[0], T, -1).type(torch.FloatTensor).cuda(gpu_id)
        # coeff, Dictionary = model.l1.forward(inputFeat)


        # covariance = np.cov(coeff.squeeze(0).cpu().numpy())
        key_list = list(np.linspace(0, T-1, T))

        """""
        Cr, covCr, covariance = get_Cr_CoCr(Dictionary, inputFeat, coeff, key_list)

        Cov_Coeff.append(np.expand_dims(covCr, 0))
        Coeff_LS.append(np.expand_dims(Cr, 0))
        Covariance.append(np.expand_dims(covariance, 0))
        C_FISTA.append(np.expand_dims(coeff.squeeze(0).cpu().numpy(), 0))
        """""
        t0 = time.time()
        'Fista solution'
        outFeat = model.forward(inputFeat)
        _, c_fist = get_recover_fista(Dictionary, inputFeat, key_list, gpu_id)

        'LS solution'
        # LScoeff = torch.matmul(a, inputFeat)
        # Cr = torch.Tensor(Cr).unsqueeze(0).cuda(gpu_id)
        # outFeat = torch.matmul(Dictionary,LScoeff)
        # _, c = get_recover(Dictionary, inputFeat, key_list)
        #
        'Cov solution'
        # covCr = torch.Tensor(covCr).unsqueeze(0).cuda(gpu_id)

        # covCr = torch.matmul(b, inputFeat)
        # outFeat = torch.matmul(Dictionary, covCr)

        # covariance = torch.Tensor(meanCov).cuda(gpu_id)
        # _, c_cov = get_recover_withCov(Dictionary, inputFeat,key_list, 0.0001, covariance, gpu_id)

        endtime = time.time() - t0
        Time.append(endtime)
        error = torch.norm(outFeat-inputFeat)
        Error.append(error.data.item())

        # print('check')
        # var = np.cov(coeff.cpu().numpy())
        # Covariance.append(np.expand_dims(var, 0))
        # COEFF.append(coeff.cpu().numpy())


        'plot hist'
        # if i == 2:
        #     covC = np.concatenate(Cov_Coeff)[:, :, 0:3]
        #     lsC = np.concatenate(Coeff_LS)[:, :, 0:3]
        #     fistC = np.concatenate(C_FISTA)[:, :, 0:3]
        #     bins = np.linspace(-0.5, 1.0, 20)
        #     plt.hist(covC.reshape(1,-1).squeeze(0), bins, alpha=0.5, label='Cr_cov')
        #     plt.hist(lsC.reshape(1,-1).squeeze(0), bins, alpha=0.5, label='Cr_ls')
        #     plt.hist(fistC.reshape(1, -1).squeeze(0), bins, alpha=0.5, label='Cr_fista')
        #     plt.legend(loc='upper right')
        #     plt.savefig('./matfile/compare_crCov_crLS_crFista.png')
        #     plt.close()


print('Average Time/sequence:', statistics.mean(Time), 'reconstruction error:', np.mean(np.asarray(Error)))
print('total time:', t_init2_end + statistics.mean(Time))


# print('check')
"""""
CovCoeff = np.concatenate((Cov_Coeff))
LSCoeff = np.concatenate((Coeff_LS))
FistCoeff = np.concatenate((C_FISTA))
meanCov = np.mean(Covariance, axis=0)

data = {'CovCoeff':Cov_Coeff, 'LSCoeff':LSCoeff, 'FistCoeff':FistCoeff, 'meanCov': meanCov}
scipy.io.savemat('./matfile/CovarianceMatrix_fullSequence.mat', mdict=data)
"""
print('done')

