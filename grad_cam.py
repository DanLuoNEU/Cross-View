from cam_utils import GradCamModel, load_data
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage.io import imread
from skimage.transform import resize

Alpha = 0.1
lam1 = 2
lam2 = 1
gpu_id = 0
num_class = 10
fusion = False

Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()

def vis_img(img, heatmap):

    heatmap_max = heatmap.max(axis = 0)[0]
    heatmap /= heatmap_max
    heatmap = resize(heatmap,(224,224),preserve_range=True)

    cmap = mpl.cm.get_cmap('jet', 256)
    heatmap2 = cmap(heatmap,alpha = 0.2)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize = (5,5))
    ax1.imshow((img))
    ax2.imshow(heatmap)
    plt.show()

def overlay(inp_imgs, acts, grads):

    # act: 1, 256, 7, 14, 14
    # grads: 1, 256, 7, 14, 14
    # inp_imgs: 1, 21, 3, 224, 224

    inp_imgs = inp_imgs.cpu().detach().numpy()
    num_images = inp_imgs.shape[1]
    act = torch.mean(acts, 2, keepdim=True)
    grad = torch.mean(grads, 2, keepdim=True)

    act = torch.squeeze(act) #(256, 14, 14)
    grad = torch.squeeze(grad) #(256, 14, 14)

    pooled_grads = torch.mean(grad, dim=[1,2]).detach().cpu()

    for i in range(act.shape[0]):
        act[i,:,:] += pooled_grads[i]

    for num in range(num_images):

        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        print('inp_img ', inp_img.shape)
        inp_img = inp_img.transpose((1, 2, 0))
        heatmap = torch.mean(act, dim = 0).squeeze()
        print('after inp_img ', inp_img.shape)
        print('after heatmap ', heatmap.shape)
        vis_img(inp_img, heatmap)

def vis_att_map():

    gcmodel = GradCamModel().to('cuda:0')
    test_loader = load_data()

    for s, sample in enumerate(test_loader):

        'Multi'
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        input_images = sample['input_image']
        inputImage = sample['input_image'].float().cuda(gpu_id)

        t = inputSkeleton.shape[2]
        y = sample['action'].data.item()

        print('inputSkeleton shape: ', inputSkeleton.shape)
        print('inputImage shape: ', inputImage.shape)
        
        for i in range(0, inputSkeleton.shape[1]):
            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            inputImg_clip = inputImage[:,i, :, :, :]
        
            if fusion:
                label_clip, _, _ = gcmodel.net.dynamicsClassifier(input_clip, t) # two stream, dynamcis branch
            else:
                label_clip, b, outClip_v,  act = gcmodel(input_clip, inputImg_clip, t, fusion)
                

            label = label_clip
            clipMSE = mseLoss(outClip_v, input_clip)
            bi_gt = torch.zeros_like(b)
            clipBI = L1loss(b, bi_gt)

            y = torch.tensor([y]).cuda(gpu_id)
            
            loss = lam1 * Criterion(label, y) + Alpha * clipBI + lam2 * clipMSE
            loss.backward()

            pred = torch.argmax(label).data.item()
            act = act.detach().cpu() #[1, 256, 7, 14, 14]
            grads = gcmodel.get_act_grads().detach().cpu() #[1, 256, 7, 14, 14]
            
            print('act: ', act.shape)
            print('input_images shape ', inputImg_clip.shape)
            overlay(inputImg_clip, act, grads)

if __name__ == '__main__':

    vis_att_map()