from cam_utils import load_model, load_data, load_net
import torch
from grad_cam_utils import *

Alpha = 0.1
lam1 = 2
lam2 = 1
gpu_id = 0
num_class = 10
fusion = False

Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()

count = 0
pred_cnt = 0
Acc = []
classLabel = [[] for i in range(0, num_class)]
classGT = [[] for i in range(0, num_class)]


def vis_att_map():

    test_loader = load_data()

    stateDict = load_model()
    net = load_net(num_class, stateDict)
    heatmap_layer = net.RGBClassifier.featureExtractor

    with torch.no_grad():

        for s, sample in enumerate(test_loader):

            'Multi'
            inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
            input_images = sample['input_image']
            inputImage = sample['input_image'].float().cuda(gpu_id)

            t = inputSkeleton.shape[2]
            y = sample['action'].data.item()
            label = torch.zeros(inputSkeleton.shape[1], num_class)
            clipBI = torch.zeros(inputSkeleton.shape[1])
            clipMSE = torch.zeros(inputSkeleton.shape[1])
            
            print('inputSkeleton shape: ', inputSkeleton.shape)
            print('inputImage shape: ', inputImage.shape)
            
            for i in range(0, inputSkeleton.shape[1]):
                input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
                inputImg_clip = inputImage[:,i, :, :, :]
            
                if fusion:
                    label_clip, _, _ = net.dynamicsClassifier(input_clip, t) # two stream, dynamcis branch
                else:
                    
                    # input_clip = input_clip.requires_grad_(True)
                    # inputImg_clip = inputImg_clip.requires_grad_(True)
                    label_clip = net.RGBClassifier(inputImg_clip)
                    grad_cam(net, inputImg_clip, label, heatmap_layer, truelabel=y)
                    
                label[i] = label_clip
                #clipMSE[i] = mseLoss(outClip_v, input_clip)

                #bi_gt = torch.zeros_like(b).cuda(gpu_id)
                #clipBI[i] = L1loss(b, bi_gt)

            label = torch.mean(label, 0, keepdim=True)
            # clipMSE = torch.mean(clipMSE, 0, keepdim=True)
            # clipBI = torch.mean(clipBI, 0, keepdim=True)
            pred = torch.argmax(label).data.item()

            #grad_cam(net, inputImg_clip, label, heatmap_layer, truelabel=y)

            count += 1
            classGT[y].append(y)
            if pred == y:
                classLabel[y].append(pred)
                pred_cnt += 1
                
            print('sample:',s, 'gt:', y, 'pred:', pred)

        Acc = pred_cnt/count

    print('Acc:', Acc, 'total sample:', count, 'correct preds:', pred_cnt)
    print('done')

if __name__ == '__main__':

    vis_att_map()