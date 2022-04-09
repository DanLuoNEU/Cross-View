from torch.optim import lr_scheduler
from torch.backends import cudnn
from torch.utils.data import DataLoader
from utils import *
from modelZoo.networks import *
from modelZoo.actHeat import *
from dataset.NUCLA_ViewProjection_trans import *
from dataset.NTU_viewProjection import *
from modelZoo.Unet import viewTransformer
from matplotlib import pyplot as plt

gpu_id = 3
num_workers = 2
PRE = 0
dataset = 'NUCLA'
#
modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'

modelPath = os.path.join(modelRoot, dataset, 'viewTransformer_heatmaps/')
stateDict = torch.load(os.path.join(modelPath, '200.pth'))['state_dict']


if dataset == 'NUCLA':
    T = 25
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_res'
    testSet = NUCLA_viewProjection(root_skeleton=root_skeleton, root_list=path_list, phase='val', cam='1,2', T=T,
                                   target_view='view_1', project_view='view_2', test_view='view_3')
    # #
    testloader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=num_workers)

in_channel = T
net = viewTransformer(in_channel=in_channel).cuda(gpu_id)
net.load_state_dict(stateDict)
net.eval()

with torch.no_grad():
    Error = []
    for i, sample in enumerate(testloader):
        target_view_heat = sample['target_view_heat']  # view_1
        project_view_heat = sample['project_view_heat']  # view_2

        target_view = target_view_heat.cuda(gpu_id).float().squeeze(0).permute(1, 0, 2, 3)
        project_view = project_view_heat.cuda(gpu_id).float().squeeze(0).permute(1, 0, 2, 3)
        out_heatmap = net(project_view)

        'load rgb image'
        target_view_rgb = sample['target_view_image'].squeeze(0).numpy()
        project_view_rgb = sample['project_view_image'].squeeze(0).numpy()
        idex1 = sample['target_info']['time_offset']
        idex2 = sample['project_info']['time_offset']

        if target_view_rgb.shape[0] >= len(idex1):
            imgSeq_view1 = target_view_rgb[idex1]
        else:
            Tadd = abs(target_view_rgb.shape[0] - len(idex1))
            last = np.expand_dims(target_view_rgb[-1],0)
            copyLast = np.repeat(last, Tadd, 0)
            imgSeq_view1 = np.concatenate((target_view_rgb, copyLast), 0)

        if project_view_rgb.shape[0] >= len(idex2):
            imgSeq_view2 = project_view_rgb[idex2]
        else:
            Tadd = abs(project_view_rgb.shape[0] - len(idex1))
            last = np.expand_dims(project_view_rgb[-1], 0)
            copyLast = np.repeat(last, Tadd, 0)
            imgSeq_view2 = np.concatenate((project_view_rgb, copyLast), 0)

        # imgSeq_view2 = project_view_rgb.squeeze(0).numpy()
        # if imgSeq_view2


        'visualization'
        folderRoot = './vis/UCLA/heatmap/' + sample['target_info']['name_sample'][0]
        if not os.path.exists(folderRoot):
            os.makedirs(folderRoot)
        heatmapGT = np.max(target_view.permute(1, 0, 2, 3).cpu().numpy(), axis=1)
        heatmapTR = np.max(out_heatmap.permute(1, 0, 2, 3).cpu().numpy(), axis=1)
        heatmapPR = np.max(project_view.permute(1, 0, 2, 3).cpu().numpy(), axis=1)


        trFolder = folderRoot + '/TR'
        gtFolder = folderRoot + '/GT'
        prFolder = folderRoot + '/PR'
        v1Folder = folderRoot + '/view1'
        v2Folder = folderRoot + '/view2'

        if not os.path.exists(trFolder):
            os.makedirs(trFolder)

        if not os.path.exists(gtFolder):
            os.makedirs(gtFolder)

        if not os.path.exists(prFolder):
            os.makedirs(prFolder)

        if not os.path.exists(v1Folder):
            os.makedirs(v1Folder)
        if not os.path.exists(v2Folder):
            os.makedirs(v2Folder)

        for t in range(0, T):
            filename_tr = 'view1_TR_'+str(t)+'.jpg'
            filename_gt = 'view1_GT_'+str(t)+'.jpg'
            filename_pr = 'view2_PR_'+str(t)+'.jpg'

            filename_v1 = 'view1_rgb_'+str(t)+'.jpg'
            filename_v2 = 'view2_rgb_'+str(t)+'.jpg'

            # plt.imsave(os.path.join(trFolder, filename_tr), heatmapTR[t], cmap='seismic', format='jpg')
            # plt.imsave(os.path.join(gtFolder, filename_gt), heatmapGT[t], cmap='seismic', format='jpg')
            # plt.imsave(os.path.join(prFolder, filename_pr), heatmapPR[t], cmap='seismic', format='jpg')
            cv2.imwrite(os.path.join(v1Folder, filename_v1), imgSeq_view1[t])
            cv2.imwrite(os.path.join(v2Folder, filename_v2), imgSeq_view2[t])


        error = torch.norm(out_heatmap - target_view)
        Error.append(error.data.item())

print('projection error:', np.mean(np.array(Error)))
print('done')