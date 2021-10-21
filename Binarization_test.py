from torch.utils.data import DataLoader
from utils import *
import scipy.io
from modelZoo.networks import *
import torch.nn
from modelZoo.sparseCoding import sparseCodingGenerator
from modelZoo.actHeat import *
from modelZoo.BinaryCoding import *
from dataset.NUCLA_ViewProjection_trans import *
from modelZoo.DyanOF import *
from dataset.NTU_viewProjection import *

'parameters'
gpu_id = 2
num_workers = 4
PRE = 0
dataset = 'NUCLA'
dataType = '2D'
clip = 'Multi'
T = 36

N = 40*2
map_location = torch.device(gpu_id)
modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
modelPath = modelRoot + dataset + '/newDYAN/BinarizeSparseCode_hardGumbel_v2/'
stateDict = torch.load(os.path.join(modelPath, '10.pth'), map_location=map_location)['state_dict']
Drr = stateDict['sparseCoding.rr']
Dtheta = stateDict['sparseCoding.theta']


net = binarizeSparseCode(num_binary=2*N+1, Drr=Drr, Dtheta=Dtheta, Inference=True, gpu_id=gpu_id)
net.load_state_dict(stateDict)

net.cuda(gpu_id)

# P,Pall = gridRing(N)
# Drr = abs(P)
# Drr = torch.from_numpy(Drr).float()
# Dtheta = np.angle(P)
# Dtheta = torch.from_numpy(Dtheta).float()
#
#
# Encoder = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)
# Encoder.cuda(gpu_id)
#
# binaryCoding = binaryCoding(num_binary=N+1).cuda(gpu_id)

# saveModel = os.path.join(modelRoot, dataset, '/BinarizeSparseCode')
# modelPath = modelRoot + dataset + '/BinarizeSparseCode_m32A1/'
# stateDict = torch.load(os.path.join(modelPath, '10.pth'))['state_dict']  # BinarizeSparseCode100.pth'
# binaryCoding.load_state_dict(stateDict)
# binaryCoding.eval()

# net = binarizeSparseCode(num_binary=N+1, Drr=Drr, Dtheta=Dtheta, PRE=PRE, gpu_id=gpu_id)
# modelPath = modelRoot + dataset + '/BinarizeSparseCode_m32A1/'
# stateDict = torch.load(os.path.join(modelPath, '150.pth'))['state_dict']
# net.load_state_dict(stateDict)
# net.cuda(gpu_id)
# net.eval()

if dataset == 'NUCLA':
    num_class = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    valSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T,
                                  target_view='view_2',
                                  project_view='view_1', test_view='view_3')
    valloader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=num_workers)

Error = []
with torch.no_grad():
    for i, sample in enumerate(valloader):
        # skeleton_v1 = sample['target_view_skeleton']['unNormSkeleton'].cuda(gpu_id)
        # skeleton_v2 = sample['project_view_skeleton']['unNormSkeleton'].cuda(gpu_id)

        # input_v1 = skeleton_v1.reshape(1, T, -1)
        # input_v2 = skeleton_v2.reshape(1, T, -1)


        # c1, _ = Encoder.forward2(input_v1, T)
        # c2, _ = Encoder.forward2(input_v2, T)

        # input_c1 = c1.reshape(1, c1.shape[1], njt, dim)
        # input_c2 = c2.reshape(1, c2.shape[1], njt, dim)

        # input_c1 = c1.permute(2, 1, 0).unsqueeze(3)
        # input_c2 = c2.permute(2, 1, 0).unsqueeze(3)
        #
        # out_b1 = binaryCoding(input_c1)
        # out_b2 = binaryCoding(input_c2)

        # if sample['cls'] == 1:
        #     print('check')

        # b1, c1 = net(input_v1, T)
        # b2, c2 = net(input_v2, T)

        # data = {'b1':out_b1.cpu().numpy(), 'b2':out_b2.cpu().numpy(), 'c1':c1.squeeze(0).permute(1,0).cpu().numpy(),
        #         'c2':c2.squeeze(0).permute(1,0).cpu().numpy()}
        # data = {'b1':b1.cpu().numpy(), 'b2':b2.cpu().numpy(), 'c1':c1.cpu().numpy(), 'c2':c2.cpu().numpy()}
        # scipy.io.savemat('./matFile/binarization_negPair_ep10.mat', mdict=data)

        # print('check')
        if sample['sample_name'][0] == 'a01_s09_e00':
            print('check')
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        t = inputSkeleton.shape[2]
        clipMSE = torch.zeros(inputSkeleton.shape[1]).cuda(gpu_id)
        for i in range(0, inputSkeleton.shape[1]):
            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            binary, coeff, dict = net(input_clip, t)
            output = torch.matmul(dict, coeff*binary)

            err = torch.norm(input_clip - output)
            clipMSE[i] = err
        error = torch.sum(clipMSE)
        Error.append(error.data.item())
    print('error:', np.mean(np.asarray(Error)))

print('done')

