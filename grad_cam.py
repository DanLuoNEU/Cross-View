import sklearn
from cam_utils import GradCamModel_RGB, GradCamModel_DYN, load_data, load_model, load_net
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from torchvision import transforms
from skimage.io import imread
from skimage.transform import resize
import cv2
import grad_cam_utils
from dataset.NUCLA_ViewProjection_trans import NUCLA_viewProjection

Alpha = 0.1
lam1 = 2
lam2 = 1
gpu_id = 0
num_class = 10
fusion = False

Criterion = torch.nn.CrossEntropyLoss()
mseLoss = torch.nn.MSELoss()
L1loss = torch.nn.L1Loss()

def stack_img(img):

    imgs = np.zeros((img.shape[0], img.shape[1], 3)).astype('float')
    imgs[:, :, 0] = img
    imgs[:, :, 1] = img
    imgs[:, :, 2] = img

    return imgs

def get_3channel(arr):

    if len(arr.shape)>2 and arr.shape[2]==3:
        return arr

    arr = arr.reshape((arr.shape[0], arr.shape[1], -1))
    arr = np.concatenate((arr, arr, arr), axis=2)

    return arr

def add_overlay(img, labels):

    disp_img = np.copy(img)
    
    disp_img = get_3channel(disp_img)

    combine_hmap = np.copy(labels)
    
    combine_hmap = cv2.resize(combine_hmap, (disp_img.shape[1], disp_img.shape[0]))
    combine_hmap = np.clip(combine_hmap*255.0/np.max(combine_hmap), 0, 255)
    combine_hmap = get_3channel(combine_hmap)

    combine_hmap = cv2.applyColorMap((combine_hmap).astype('uint8'),cv2.COLORMAP_VIRIDIS)

    disp_img = cv2.addWeighted(disp_img.astype('uint8'), 0.3, combine_hmap , 0.7,
        0)

    cv2.imshow('disp_img ', combine_hmap)
    
    cv2.imshow('vis_img ', img)
    cv2.waitKey(-1)

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
        add_overlay(inp_img, heatmap)

def vis_grad_pole(grads_org):

    grads = np.copy(grads_org)[0]
    grads = grads - np.min(grads)
    print('grads 0 min max: ', np.min(grads), np.max(grads))

    grads = grads / np.max(grads)

    print('grads 1 min max: ', np.min(grads), np.max(grads))

    grads = np.clip(grads*255, 0, 255).astype('uint8')    
    cv2.imshow('grads ', grads)
    cv2.waitKey(-1)

def get_main_joints(grads_org):

    grads = np.copy(grads_org)[0]
    max_grads = np.max(grads, axis=0)
    
    print('max_grads: ', max_grads)

def remap_image_val(org_img):

    img = org_img.copy()
    img = img.astype('float')
    img = img - np.min(img)
    img = img/np.max(img)

    return np.clip(255.0*img, 0, 255).astype('uint8')

def reproject_skel(skeletons):

    crop_size = 224
    fac = 256.0/480
    w, h = 640, 480

    skeletons[:, :] *= fac
    
    cy, cx = h*fac//2, w*fac//2
    skeletons[:, 0] -= cx - crop_size//2
    skeletons[:, 1] -= cy - crop_size//2

    return skeletons

def plot_skeleton(org_img, skeletons, resize_fac=2):

    img = np.copy(org_img)
    img = remap_image_val(img)
    skeletons = reproject_skel(skeletons)

    img = cv2.resize(img, None, fx=resize_fac, fy=resize_fac)
    skeletons[:, :] *= resize_fac
    
    for skeleton in skeletons:

        skeleton = (int(skeleton[0]), int(skeleton[1]))
        cv2.circle(img, center=skeleton, radius=1, color=(255,0,0), thickness=-1)

    return img

def vis_poles(data, inp_imgs, inp_clips, acts, grads):

    acts = acts.cpu().detach().numpy()
    grads = grads.cpu().detach().numpy()
    inp_imgs = inp_imgs.cpu().detach().numpy()
    inp_clips = inp_clips.cpu().detach().numpy()
    num_images = inp_imgs.shape[1]

    print('inp_clips shape: ', inp_clips.shape)     
    inp_clips_unnorm = np.copy(inp_clips).reshape((inp_clips.shape[0], inp_clips.shape[1], -1, 2))     

    inp_clips_unnorm = data.get_unnorm(inp_clips_unnorm)
    inp_clips_unnorm = inp_clips_unnorm.cpu().detach().numpy()

    print('acts min max: ', np.min(acts), np.max(acts))
    print('grads min max: ', np.min(grads), np.max(grads))

    acts = np.clip(acts*255.0, 0, 255)
    acts = acts.astype('uint8')

    vis_grad_pole(grads)

    for num in range(num_images):

        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        inp_skel = inp_clips_unnorm[num, :, :] #(25, 2)
        inp_img = inp_img.transpose((1, 2, 0)) #(224, 224, 3)
        
        skel_img = plot_skeleton(inp_img, inp_skel)
        cv2.imshow('inp_img ', inp_img)
        cv2.imshow('skel_img ', skel_img)
        cv2.imshow('acts ', acts[0])
        cv2.waitKey(-1)

def vis_att_map_rgb():

    gcmodel = GradCamModel_RGB().to('cuda:0')
    _, test_loader = load_data()

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

def vis_att_map_dyn():

    gcmodel = GradCamModel_DYN().to('cuda:0')
    test_dataset, test_loader = load_data()

    for s, sample in enumerate(test_loader):

        'Multi'
        inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
        input_images = sample['input_image']
        inputImage = sample['input_image'].float().cuda(gpu_id)

        t = inputSkeleton.shape[2]
        y = sample['action'].data.item()

        print('sample keys: ', sample.keys())
        print('inputSkeleton shape: ', inputSkeleton.shape)
        print('inputImage shape: ', inputImage.shape)
        
        for i in range(0, inputSkeleton.shape[1]):
            input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
            inputImg_clip = inputImage[:, i, :, :, :]
        
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
            act = act.detach().cpu() 
            grads = gcmodel.get_act_grads().detach().cpu() 
               
            print('act: ', act.shape)
            print('grads: ', grads.shape)
            print('input_images shape ', inputImg_clip.shape)
            print('input_clip ', input_clip[0], input_clip[0].shape)

            vis_poles(test_dataset, inputImg_clip, input_clip, act, grads)

if __name__ == '__main__':

    #vis_att_map_rgb()
    vis_att_map_dyn()