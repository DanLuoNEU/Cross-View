from cam_utils import GradCamModel_RGB, GradCamModel_DYN, load_data, load_model, load_net
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import grad_cam_utils
import saliency_utils

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

def visualize_saliency_rgb(inp_imgs, saliency):

    inp_imgs = inp_imgs.cpu().detach().numpy()
    saliency = saliency.cpu().detach().numpy()
    num_images = inp_imgs.shape[1]

    for num in range(num_images):
        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        sal = np.squeeze(saliency[:, num, :, :])
        inp_img = inp_img.transpose((1, 2, 0))
        
        print('inp_img ', inp_img.shape)
        print('sal ', sal.shape)
        
        sal = sal/np.max(sal)
        sal = np.clip(sal * 255.0, 0, 255.0).astype('uint8')
        cv2.imshow('inp_img ', inp_img)
        cv2.imshow('sal ', sal)
        cv2.waitKey(-1)
        
def visualize_saliency_dyn(inp_imgs, inp_clips, saliency):

    inp_clips = inp_clips.cpu().detach().numpy()
    inp_imgs = inp_imgs.cpu().detach().numpy()
    saliency = saliency.cpu().detach().numpy()
    num_clips = inp_clips.shape[1]

    print('inp_clips shape: ', inp_clips.shape)
    print('saliency shape: ', saliency.shape)

    for num in range(num_clips):
        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        inp_clip = np.squeeze(inp_clips[:, num, :])
        sal = np.squeeze(saliency[:, num, :])
        inp_img = inp_img.transpose((1, 2, 0))

        print('inp_clip ', inp_clip.shape)
        print('sal ', sal.shape)
        print('sal min max: ', np.min(sal), np.max(sal), sal)

def vis_poles(inp_imgs, acts, grads):

    acts = acts.cpu().detach().numpy()
    grads = grads.cpu().detach().numpy()
    inp_imgs = inp_imgs.cpu().detach().numpy()
    num_images = inp_imgs.shape[1]
    
    print('acts min max: ', np.min(acts), np.max(acts))
    print('grads min max: ', np.min(grads), np.max(grads))

    acts = np.clip(acts*255.0, 0, 255)
    print('acts ', acts)
    acts = acts.astype('uint8')

    for num in range(num_images):

        inp_img = np.squeeze(inp_imgs[:, num, :, :, :])
        print('inp_img ', inp_img.shape)
        inp_img = inp_img.transpose((1, 2, 0))

        cv2.imshow('inp_img ', inp_img)
        cv2.imshow('acts ', acts[0])
        cv2.waitKey(-1)

def vis_att_map_rgb():

    test_loader = load_data()
    stateDict = load_model()
    net = load_net(num_class, stateDict)

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

            inputImg_clip.requires_grad_()
            label = net.RGBClassifier(inputImg_clip)
            y = torch.tensor([y]).cuda(gpu_id)
            saliency = saliency_utils.compute_saliency_maps_rgb(inputImg_clip, y, label)
            visualize_saliency_rgb(inputImg_clip, saliency)

def vis_att_map_dyn():

    test_loader = load_data()
    stateDict = load_model()
    net = load_net(num_class, stateDict)

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

            input_clip.requires_grad_()
            label, _, _, _ = net.dynamicsClassifier(input_clip, t)
            y = torch.tensor([y]).cuda(gpu_id)
            saliency = saliency_utils.compute_saliency_maps_dyn(input_clip, y, label)
            visualize_saliency_dyn(inputImg_clip, input_clip, saliency)

if __name__ == '__main__':

    #vis_att_map_rgb()
    vis_att_map_dyn()