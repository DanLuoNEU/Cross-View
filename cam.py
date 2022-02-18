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
from lossFunction import binaryLoss

gpu_id = 1
num_workers = 4
fistaLam = 0.1
print('gpu_id: ',gpu_id)
print('num_workers: ',num_workers)
print('fistaLam: ',fistaLam)

PRE = 0
T = 36
dataset = 'NUCLA'
# dataset = 'NTU'
Alpha = 0.1
lam1 = 2
lam2 = 1
N = 80*2
Epoch = 150
dataType = '2D'
# clip = 'Single'
clip = 'Multi'
fusion = False
# fusion = True

modelRoot = '/home/balaji/Documents/code/RSL/CS_CV/Cross-View/models/'
# modelPath = modelRoot + dataset + '/CS/2S_Multi_500_openpose_lam1051/'
# modelPath = modelRoot + dataset + '/2Stream/train_t36_CV_openpose_testV3_lam1051/'
# modelPath = modelRoot + dataset + '/2Stream/multiClip_lam1051_testV2_CV/'
# modelPath = modelRoot + dataset + '/DYOnly_Multi_CV/'
modelPath = modelRoot + dataset + '/0216/CV_dynamicsStream_fista01_reWeighted_sqrC_T72/'
map_location = torch.device(gpu_id)
stateDict = torch.load(os.path.join(modelPath, '0.pth'), map_location=map_location)['state_dict']
Drr = stateDict['dynamicsClassifier.sparseCoding.rr']
Dtheta = stateDict['dynamicsClassifier.sparseCoding.theta']

# Drr = stateDict['sparseCoding.rr']
# Dtheta = stateDict['sparseCoding.theta']

if dataset == 'NUCLA':
    num_class = 10
    path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    'CS:'
    # testSet = NUCLAsubject(root_list=path_list, dataType=dataType, clip=clip, phase='test', T=T)
    # testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=num_workers)


    'CV:'
    testSet = NUCLA_viewProjection(root_list=path_list, dataType=dataType, clip=clip, phase='test', cam='1,2', T=T,
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

    # testSet = NTURGBDsubject(root_skeleton, nanList, dataType=dataType, clip=clip, phase='test', T=36)
    # testloader = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    'CV:'
    testSet = NTURGBD_viewProjection(root_skeleton, root_list="/data/Yuexi/NTU-RGBD/list/", nanList=nanList, dataType= dataType, clip=clip,
                                phase='test', T=36, target_view='C002', project_view='C001', test_view='C003')
    testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=num_workers)

# net = classificationWSparseCode(num_class=10, Npole=2*N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2,gpu_id=gpu_id).cuda(gpu_id)
# net = Fullclassification(num_class=num_class, Npole=2*N+1, num_binary=2*N+1, Drr=Drr, Dtheta=Dtheta, dim=2, dataType=dataType, Inference=True, gpu_id=gpu_id, fistaLam=fistaLam).cuda(gpu_id)
kinetics_pretrain = './pretrained/i3d_kinetics.pth'
net = twoStreamClassification(num_class=num_class, Npole=(2*N+1), num_binary=(2*N+1), Drr=Drr, Dtheta=Dtheta,
                                  dim=2, gpu_id=gpu_id, dataType=dataType, fistaLam=fistaLam,  kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

# net = RGBAction(num_class=num_class, kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
net.load_state_dict(stateDict)
net.eval()

# from torchsummary import summary
# summary(net)
# print('net: ',net)

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net.RGBClassifier.layer1.register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())

for name, param in net.named_parameters():
    print(name, param.data.cpu().detach().numpy().shape)

for i in range(len(params)):
    print('param ' + str(i), params[i].data.cpu().detach().numpy().shape)
    
weight_softmax = np.squeeze(params[-4].data.cpu().detach().numpy())
print('weight_softmax shape: ', weight_softmax.shape)

count = 0
pred_cnt = 0
Acc = []
classLabel = [[] for i in range(0, num_class)]
classGT = [[] for i in range(0, num_class)]

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256

    size_upsample = (224, 224)
    _, bz, nc, h, w = feature_conv.shape
    print('feature_conv shape: ', feature_conv.shape)
    output_cam = []

    cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
        
    return output_cam

def display_images(input_img_batch):

    input_img_batch = np.transpose(input_img_batch, (1, 3, 4, 2, 0)).squeeze(axis=-1)
    num_images = input_img_batch.shape[0]

    for num_image in range(num_images):
        img = input_img_batch[num_image].cpu().detach().numpy()
        cv2.imshow('image ', img)
        cv2.waitKey(-1)

    #Result
    img = input_img_batch[0].cpu().detach().numpy()
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imshow('result ', result)
    cv2.waitKey(-1)

with torch.no_grad():

    for s, sample in enumerate(testloader):
        # input = sample['test_view_multiClips'].float().cuda(gpu_id)
        'Multi'
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        input_images = sample['input_image']

        #inputSkeleton = sample['project_skeleton'].float().cuda(gpu_id)
        inputImage = sample['input_image'].float().cuda(gpu_id)

        'Single clip'
        # inputSkeleton = sample['input_skeleton']['normSkeleton'].unsqueeze(0).float().cuda(gpu_id)
        # inputImage = sample['input_image'].unsqueeze(0).float().cuda(gpu_id)
        # input_images = sample['input_image'].unsqueeze(0)

        t = inputSkeleton.shape[2]
        y = sample['action'].data.item()
        label = torch.zeros(inputSkeleton.shape[1], num_class)
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
        
        print('inputSkeleton shape: ', inputSkeleton.shape)
        print('inputImage shape: ', inputImage.shape)
        
        for i in range(0, inputSkeleton.shape[1]):
            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            # input_clip = input_clip[:, :, 1].unsqueeze(2)
            inputImg_clip = inputImage[:,i, :, :, :]
            # label_clip, _, _, _ = net(input_clip, t) # DY+BI+CL
            # label_clip, _ = net(input_clip, t) # DY+CL
            
            if fusion:
                label_clip, _, _ = net.dynamicsClassifier(input_clip, t) # two stream, dynamcis branch
                # label_clip = net.RGBClassifier(inputImg_clip)
            else:
                label_clip, _, _ = net(input_clip, inputImg_clip, t, fusion)
                # label_clip, _ = net(input_clip, t)  # DY + CL
                # label_clip = net(inputImg_clip)

            label[i] = label_clip
            print('label_clip: ', label_clip)
            input_img_batch = input_images[:, 0, :, :, :]
            print('len: ', len(features_blobs))
            CAMs = returnCAM(features_blobs[0], weight_softmax, [label_clip])
            #display_images(input_img_batch)

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
print('done')