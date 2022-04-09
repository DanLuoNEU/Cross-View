from torch.utils.data import DataLoader
from utils import *
import scipy.io
from modelZoo.networks import *
import torch.nn
from modelZoo.sparseCoding import sparseCodingGenerator
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from modelZoo.DyanOF import *
from dataset.NTU_viewProjection import *
import json

gpu_id = 3
num_workers = 2
PRE = 0
dataset = 'NUCLA'
# dataset = 'posePose'
def binarizedSparseCode(sparseCode):
    positiveC = (torch.pow(sparseCode, 2) + 1e-6) / (torch.sqrt(torch.pow(sparseCode, 2)) + 1e-6)
    sparseCode = torch.tanh(10*positiveC)

    return sparseCode
def getAttentionHeat(heatmap, gpu_id):
    softmax = torch.nn.Softmax2d()
    heatmapMax = softmax(heatmap)
    Grid = torch.randn(heatmap.shape).cuda(gpu_id)
    normGrid = 2*((Grid-torch.min(Grid))/(torch.max(Grid)-torch.min(Grid))) -1
    heatAttention = torch.matmul(normGrid, heatmapMax)

    return heatAttention,normGrid,heatmapMax

def getJsonData(fileRoot, folder):
    skeleton = []
    allFiles = os.listdir(os.path.join(fileRoot, folder))
    allFiles.sort()
    for i in range(0, len(allFiles)):
        with open(os.path.join(fileRoot,folder, allFiles[i])) as f:
            data = json.load(f)
        temp = data['people'][0]['pose_keypoints_2d']
        pose = np.expand_dims(np.asarray(temp).reshape(25, 3)[:,0:2], 0)
        skeleton.append(pose)

    skeleton = np.concatenate((skeleton))

    return torch.tensor(skeleton).type(torch.FloatTensor)




#
# modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
#
# modelPath = os.path.join(modelRoot, dataset, 'viewTransformer_heatmaps/')
# stateDict = torch.load(os.path.join(modelPath, '200.pth'))['state_dict']
N = 40*4
Epoch = 40

P,Pall = gridRing(N)
Drr = abs(P)
Drr = torch.from_numpy(Drr).float()
Dtheta = np.angle(P)
Dtheta = torch.from_numpy(Dtheta).float()
net = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)
net.cuda(gpu_id)


T = 25
if dataset == 'NUCLA':

    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_res'
    dataSet = NUCLA_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='train', cam='1,2', T=T,
                                   target_view='view_1', project_view='view_2', test_view='view_3')
    # #
    dataloader = DataLoader(dataSet, batch_size=1, shuffle=False, num_workers=num_workers)


# fileRoot = './forOpenPose/a11_s04_e00'
#
# skeleton_v1 = getJsonData(fileRoot, 'view1')
# skeleton_v2 = getJsonData(fileRoot, 'view2')
#
# input_v1 = skeleton_v1.reshape(T, -1).unsqueeze(0).cuda(gpu_id)
# input_v2 = skeleton_v2.reshape(T, -1).unsqueeze(0).cuda(gpu_id)
#
# c1, _ = net.forward2(input_v1, T)
# c2, _ = net.forward2(input_v2, T)
#
# data = {'c1': c1.cpu().squeeze(0).numpy(), 'c2': c2.cpu().squeeze(0).numpy()}
# scipy.io.savemat('./matFile/coeff_compare_openpose_a11_fis1.mat', mdict=data)



# print('check')



with torch.no_grad():

    Error = []
    for i, sample in enumerate(dataloader):
        skeleton_v1 = sample['target_view_skeleton']['unNormSkeleton'].cuda(gpu_id)
        skeleton_v2 = sample['project_view_skeleton']['unNormSkeleton'].cuda(gpu_id)

        input_v1 = skeleton_v1.reshape(1, T, -1)
        input_v2 = skeleton_v2.reshape(1, T, -1)
        # target_view_heat = sample['target_view_heat']  # view_1
        # project_view_heat = sample['project_view_heat']

        njt = 19
        # input_v1 = target_view_heat[:,:,njt].reshape(1, T, -1).type(torch.FloatTensor).cuda(gpu_id)
        # input_v2 = project_view_heat[:,:, njt].reshape(1,T, -1).type(torch.FloatTensor).cuda(gpu_id)
        # input_v1 = target_view_heat[:, :, njt].type(torch.FloatTensor).cuda(gpu_id)
        # input_v2 = project_view_heat[:, :, njt].type(torch.FloatTensor).cuda(gpu_id)

        # heatAttention_v1, normGrid1, heatSoftmax_v1 = getAttentionHeat(input_v1, gpu_id)
        # heatAttention_v2, normGrid2, heatSoftmax_v2 = getAttentionHeat(input_v2, gpu_id)

        # v1 = heatAttention_v1.reshape(1, T, -1)
        # v2 = heatAttention_v2.reshape(1, T, -1)

        c1, D1 = net.forward2(input_v1, T)
        c2, D2 = net.forward2(input_v2, T)

        # c1_bi = binarizedSparseCode(c1)
        # c2_bi = binarizedSparseCode(c2)


        # data = {'c1':c1.cpu().squeeze(0).numpy(), 'c2':c2.cpu().squeeze(0).numpy(),
        #         'c1_bi':c1_bi.cpu().squeeze(0).numpy(), 'c2_bi':c2_bi.cpu().squeeze(0).numpy()}

        """""
        data = {'c1': c1.cpu().squeeze(0).numpy(), 'c2': c2.cpu().squeeze(0).numpy()}
        scipy.io.savemat('./matFile/coeff_compare_heatAttention_njt19.mat', mdict=data)


             
        'visualization'
        folderRoot = '/home/yuexi/Documents/Cross-view/vis/UCLA/heatmap/singleHeat1'
        if not os.path.exists(folderRoot):
            os.makedirs(folderRoot)
        v1_heat = input_v1.squeeze(0).reshape(T, 64, 64).cpu().numpy()
        v2_heat = input_v2.squeeze(0).reshape(T, 64, 64).cpu().numpy()
        v1Folder = folderRoot + '/view1'
        v2Folder = folderRoot + '/view2'
        if not os.path.exists(v1Folder):
            os.makedirs(v1Folder)
        if not os.path.exists(v2Folder):
            os.makedirs(v2Folder)

        for t in range(0, T):
            filename_v1 = 'view1_hm' + str(t) + '.jpg'
            filename_v2 = 'view2_hm' + str(t) + '.jpg'

            plt.imsave(os.path.join(v1Folder, filename_v1), v1_heat[t], cmap='seismic', format='jpg')
            plt.imsave(os.path.join(v2Folder, filename_v2), v2_heat[t], cmap='seismic', format='jpg')
        
        'check attention heatmap'
        folderRoot = '/home/yuexi/Documents/Cross-view/vis/UCLA/heatmap/attentionHeat'
        if not os.path.exists(folderRoot):
            os.makedirs(folderRoot)
        v1Folder = folderRoot + '/view1'
        v2Folder = folderRoot + '/view2'
        if not os.path.exists(v1Folder):
            os.makedirs(v1Folder)
        if not os.path.exists(v2Folder):
            os.makedirs(v2Folder)
        t = 0
        'view 1'
        plt.imsave(os.path.join(v1Folder, 'originalHeat.jpg'), input_v1[0,t].cpu().numpy(), cmap='seismic', format='jpg')
        plt.imsave(os.path.join(v1Folder, 'softmaxHeat.jpg'), heatSoftmax_v1[0,t].cpu().numpy(), cmap='seismic', format='jpg')
        plt.imsave(os.path.join(v1Folder, 'normGrid.jpg'), normGrid1[0,t].cpu().numpy(), cmap='seismic', format='jpg')
        plt.imsave(os.path.join(v1Folder, 'heatAttention.jpg'), heatAttention_v1[0,t].cpu().numpy(), cmap='seismic', format='jpg')

        'view 2'
        plt.imsave(os.path.join(v2Folder, 'originalHeat.jpg'), input_v2[0, t].cpu().numpy(), cmap='seismic',format='jpg')
        plt.imsave(os.path.join(v2Folder, 'softmaxHeat.jpg'), heatSoftmax_v2[0, t].cpu().numpy(), cmap='seismic',format='jpg')
        plt.imsave(os.path.join(v2Folder, 'normGrid.jpg'), normGrid2[0, t].cpu().numpy(), cmap='seismic', format='jpg')
        plt.imsave(os.path.join(v2Folder, 'heatAttention.jpg'), heatAttention_v2[0, t].cpu().numpy(), cmap='seismic',format='jpg')
        
        """""
        print('check')




