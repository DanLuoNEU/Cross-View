from torch.utils.data import DataLoader
from utils import *
import scipy.io
from modelZoo.networks import *
from torch.optim import lr_scheduler
from scipy.spatial import distance
import torch.nn
from modelZoo.sparseCoding import sparseCodingGenerator
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
gpu_id = 1
num_workers = 4
PRE = 0
dataset = 'NUCLA'

N = 40*2
Epoch = 150
T = 36
dataType = '2D'
clip = 'Single'


modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
modelPath = modelRoot + dataset + '/newDYAN/sparseCode_NUCLA_T36_fist001_openPose/'
stateDict = torch.load(os.path.join(modelPath, '50.pth'))['state_dict']

Drr = stateDict['rr']
Dtheta = stateDict['theta']

net = DyanEncoder(Drr, Dtheta, lam=0.01, gpu_id=gpu_id).cuda(gpu_id)


if dataset == 'NUCLA':
    num_class = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    valSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='2,1', T=T,
                                  target_view='view_2',
                                  project_view='view_1', test_view='view_3')
    valloader = DataLoader(valSet, batch_size=1, shuffle=True, num_workers=num_workers)




with torch.no_grad():
    Error = []
    for i, sample in enumerate(valloader):
        skeleton = sample['input_skeleton']['normSkeleton'].float().cuda(gpu_id)
        t = skeleton.shape[1]
        input = skeleton.reshape(1, t, -1)

        coeff, dict = net(input, t)
        output = torch.matmul(dict, coeff)

        error = torch.norm(input - output)
        Error.append(error.data.item())

        # y = sample['cls'].data.item()
        """""
        if y == 0:
            data = {'view1Seq':sample['target_info']['name_sample'],'c1': c1.cpu().numpy(),'skeleton_v1':sample['target_view_skeleton']['unNormSkeleton'].squeeze(0).numpy(),
                    'view2Seq':sample['project_info']['name_sample'], 'c2': c2.cpu().numpy(),'skeleton_v2':sample['project_view_skeleton']['unNormSkeleton'].squeeze(0).numpy()}
            file_name = './matFile/' + dataset + '_sparseCode_' + 'cls'+ str(y) + '_trainSet_v2.mat'
            scipy.io.savemat(file_name, mdict=data)

            'load rgb image'
            target_view_rgb = sample['target_view_image'].squeeze(0).numpy()
            project_view_rgb = sample['project_view_image'].squeeze(0).numpy()
            idex1 = sample['target_info']['time_offset']
            idex2 = sample['project_info']['time_offset']

            if target_view_rgb.shape[0] >= len(idex1):
                imgSeq_view1 = target_view_rgb[idex1]
            else:
                Tadd = abs(target_view_rgb.shape[0] - len(idex1))
                last = np.expand_dims(target_view_rgb[-1], 0)
                copyLast = np.repeat(last, Tadd, 0)
                imgSeq_view1 = np.concatenate((target_view_rgb, copyLast), 0)

            if project_view_rgb.shape[0] >= len(idex2):
                imgSeq_view2 = project_view_rgb[idex2]
            else:
                Tadd = abs(project_view_rgb.shape[0] - len(idex1))
                last = np.expand_dims(project_view_rgb[-1], 0)
                copyLast = np.repeat(last, Tadd, 0)
                imgSeq_view2 = np.concatenate((project_view_rgb, copyLast), 0)
            'visualization'
            v1_folder = './vis/UCLA/RGB/PositivePair/' + sample['target_info']['name_sample'][0]
            v2_folder = './vis/UCLA/RGB/PositivePair/' + sample['project_info']['name_sample'][0]
            if not os.path.exists(v1_folder):
                os.makedirs(v1_folder)

            if not os.path.exists(v2_folder):
                os.makedirs(v2_folder)
            for t in range(0, T):
                filename_v1 = 'view1_rgb_' + str(t) + '.jpg'
                filename_v2 = 'view2_rgb_' + str(t) + '.jpg'
                cv2.imwrite(os.path.join(v1_folder, filename_v1), imgSeq_view1[t])
                cv2.imwrite(os.path.join(v2_folder, filename_v2), imgSeq_view2[t])

            print('check')
        """
        # iter+=1

print('done')