# 11/04/2020, Dan
# This is used to read in skeleton experiment list
# Northeastern-UCLA 3D Action Recognition Dataset
# If T =0 then input the whole sequence

import os
import math
import numpy as np
import pickle
import random
import torch
import torch.utils.data as tudata
from torchvision import transforms
from PIL import Image,ImageFilter,ImageEnhance

from utils import Gaussian, DrawGaussian
"""
   Hip_Center = 1;   Spine = 2;       Shoulder_Center = 3; Head = 4;           Shoulder_Left = 5;
   Elbow_Left = 6;   Wrist_Left = 7;  Hand_Left = 8;       Shoulder_Right = 9; Elbow_Right = 10;
   Wrist_Right = 11; Hand_Right = 12; Hip_Left = 13;       Knee_Left = 14;     Ankle_Left = 15;
   Foot_Left = 16;   Hip_Right = 17;  Knee_Right = 18;     Ankle_Right = 19;   Foot_Right = 20;
"""

class NUCLAsubject(tudata.Dataset):
    """Northeastern-UCLA Dataset Skeleton Dataset
        Access input skeleton sequence, GT label
        When T=0, it returns the whole 
    """
    def __init__(self, root_skeleton="/data/Dan/N-UCLA_MA_3D/skeletons_res",
                root_list="/data/Dan/N-UCLA_MA_3D/lists", phase='train', T=0):
        self.root_skeleton = root_skeleton
        self.data_root = '/data/Dan/N-UCLA_MA_3D/multiview_action'
        self.root_list = root_list
        self.T = T
        self.num_samples = 0

        self.num_action = 10
        self.action_list={'a01':0, 'a02':1, 'a03':2, 'a04':3, 'a05':4,
                        'a06':5, 'a08':6, 'a09':7, 'a11':8, 'a12':9}
        self.actions={'a01':'pick up with one hand', 'a02':"pick up with two hands", 'a03':"drop trash", 'a04':"walk around", 'a05':"sit down",
                        'a06':"stand up", 'a08':"donning", 'a09':"doffing", 'a11':"throw", 'a12':"carry"}

        # Get the list of files according to cam and phase
        self.list_samples = []
        # Compute the MEAN and STD of the dataset
        allSkeleton = []
        file_list = os.path.join(self.root_list, f"subject_{phase}.list")
        list_samples = np.loadtxt(file_list, dtype=str)
        for sample in list_samples:
            view, name_sample = sample.split(',')[0], sample.split(',')[1]
            self.list_samples.append((view, name_sample))
            skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample+'.npy'), allow_pickle=True)
            if len(skeleton.shape):
                np.asarray(skeleton)
            allSkeleton.append(skeleton)
        allSkeleton=np.concatenate(allSkeleton, 0)#((allSkeleton),0)
        self.x_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 0], 0)  # 1 x num_joint
        self.y_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 1], 0)
        self.x_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 0], 0)
        self.y_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 1], 0)
        # Print 
        print(f"=== {phase} subject: {len(self.list_samples)} samples ===")
    
    def __len__(self):
        return len(self.list_samples)

    def get_norm(self, skeleton):
        '''Get the normalization according to the skeleton
        Arguments:
            skeleton: T(8) x num_joints(25) x 2, unnormalized in original resolution
        Return:
            skeleton: T(8) x num_joints(25) x 2, normalized in original resolution
        '''
        X = skeleton[:,:,0]
        Y = skeleton[:,:,1]
        normX = (X - self.x_mean_skeleton)/self.x_std_skeleton
        normY = (Y - self.y_mean_skeleton)/self.y_std_skeleton
        normSkeleton = np.concatenate((np.expand_dims(normX,2), np.expand_dims(normY,2)), 2).astype(float) # inputLen x 15 x 2

        return normSkeleton
    
    def get_unnorm(self, normSkeleton):
        ''' Get unnormalized data back
        Argument:
            normSkeleton: T x num_joints x 2
        '''
        if len(normSkeleton.shape) == 4:
            normSkeleton = normSkeleton.squeeze(0)
        if isinstance(normSkeleton, np.ndarray):
            normSkeleton = torch.Tensor(normSkeleton).float()

        framNum = normSkeleton.shape[0]
        meanX_mat = torch.FloatTensor(self.x_mean_skeleton).repeat(framNum, 1)   # inputLen x 15
        meanY_mat = torch.FloatTensor(self.y_mean_skeleton).repeat(framNum, 1)
        stdX_mat = torch.FloatTensor(self.x_std_skeleton).repeat(framNum, 1)
        stdY_mat = torch.FloatTensor(self.y_std_skeleton).repeat(framNum, 1)

        X = normSkeleton[:,:,0]  # inputLen x 15
        Y = normSkeleton[:,:,1]  # inputLen x 15
        unNormX = X * stdX_mat + meanX_mat
        unNormY = Y * stdY_mat + meanY_mat
        unNormSkeleton = torch.cat((unNormX.unsqueeze(2), unNormY.unsqueeze(2)), 2)

        return unNormSkeleton

    def pose_to_heatmap(self, poses, image_size, outRes):
        ''' Pose to Heatmap
        Argument:
            joints: T x njoints x 2
        Return:
            heatmaps: T x 64 x 64
        '''
        GaussSigma = 1

        T = poses.shape[0]
        H = image_size[0]
        W = image_size[1]
        heatmaps = []
        for t in range(0, T):
            pts = poses[t]  # njoints x 2
            out = np.zeros((pts.shape[0], outRes, outRes))

            for i in range(0, pts.shape[0]):
                pt = pts[i]
                if pt[0] == 0 and pt[1] == 0:
                    out[i] = np.zeros((outRes, outRes))
                else:
                    newPt = np.array([outRes * (pt[0] / W), outRes * (pt[1] / H)])
                    out[i] = DrawGaussian(out[i], newPt, GaussSigma)
            out_max = np.max(out, axis=0)
            heatmaps.append(out_max)
        stacked_heatmaps = np.stack(heatmaps, axis=0)
        min_offset = -1 * np.amin(stacked_heatmaps)
        stacked_heatmaps = stacked_heatmaps + min_offset
        max_value = np.amax(stacked_heatmaps)
        if max_value == 0:
            return stacked_heatmaps
        stacked_heatmaps = stacked_heatmaps / max_value

        return stacked_heatmaps

    def get_rgb(self, view, name_sample):
        data_path = os.path.join(self.data_root, view, name_sample)
        # print(data_path)
        # fileList = np.loadtxt(os.path.join(data_path, 'fileList.txt'))
        imgId = []
        # for l in fileList:
        #     imgId.append(int(l[0]))
        # imgId.sort()

        imageList = []

        for item in os.listdir(data_path):
            if item.find('_rgb.jpg') != -1:
                id = int(item.split('_')[1])
                imgId.append(id)

        imgId.sort()

        for i in range(0, len(imgId)):
            for item in os.listdir(data_path):
                if item.find('_rgb.jpg') != -1:
                    if int(item.split('_')[1]) == imgId[i]:
                        imageList.append(item)
        # imageList.sort()
        'make sure it is sorted'

        imgSize = []
        imgSequence = []

        for i in range(0, len(imageList)):
            img_path = os.path.join(data_path, imageList[i])
            input_image = Image.open(img_path)
            imgSize.append(input_image.size)

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            img_tensor = preprocess(input_image)

            imgSequence.append(img_tensor.unsqueeze(0))

        imgSequence = torch.cat((imgSequence), 0)

        return imgSequence, imgSize

    def get_grayscale(self, view, name_sample):
        data_path = os.path.join(self.data_root, view, name_sample)
        # print(data_path)
        # fileList = np.loadtxt(os.path.join(data_path, 'fileList.txt'))
        imgId = []
        # for l in fileList:
        #     imgId.append(int(l[0]))
        # imgId.sort()

        imageList = []

        for item in os.listdir(data_path):
            if item.find('_rgb.jpg') != -1:
                id = int(item.split('_')[1])
                imgId.append(id)

        imgId.sort()

        for i in range(0, len(imgId)):
            for item in os.listdir(data_path):
                if item.find('_rgb.jpg') != -1:
                    if int(item.split('_')[1]) == imgId[i]:
                        imageList.append(item)
        # imageList.sort()
        'make sure it is sorted'

        imgSize = []
        imgSequence = []

        for i in range(0, len(imageList)):
            img_path = os.path.join(data_path, imageList[i])
            input_image = Image.open(img_path)
            imgSize.append(input_image.size)

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(64),
                transforms.Grayscale(num_output_channels=1),transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

            img_tensor = preprocess(input_image)

            imgSequence.append(img_tensor.unsqueeze(0))

        imgSequence = torch.cat((imgSequence), 0)

        return imgSequence, imgSize
    def __getitem__(self, index):
        """
        Return:
            skeletons: FloatTensor, [T, num_joints, 2]
            label_action: int, label for the action
            info: dict['sample_name', 'T_sample', 'time_offset']
        """
        view, name_sample = self.list_samples[index]
        label_action = self.action_list[name_sample[:3]] # 0-9
        # print(index, name_sample)
        # skeleton shape (T_sample, 20, 2)
        # scale of 2d skeleton, resolution 640 x 480
        skeleton=np.load(os.path.join(self.root_skeleton, view, name_sample+'.npy'), allow_pickle=True)
        T_sample, num_joints, dim = skeleton.shape

        if self.T==0:
            skeleton_input = skeleton
            # imgSequence = np.zeros((T_sample, 3, 224, 224))
            info = {'name_sample':name_sample, 'T_sample':T_sample, 'time_offset':range(T_sample)}
        else:
            stride=T_sample/self.T
            ids_sample = []
            for i in range(self.T):
                id_sample=random.randint(int(stride*i),int(stride*(i+1))-1)
                ids_sample.append(id_sample)
            
            skeleton_input = skeleton[ids_sample,:,:]
            info = {'name_sample':name_sample, 'T_sample':T_sample, 'time_offset':ids_sample}

            # Prepare Zero Images
            # imgSequence = np.zeros((self.T, 3, 224, 224))
        imgSize = (640, 480)
        # Prepare Skeleton
        normSkeleton=self.get_norm(skeleton_input)
        # Prepare heatmaps
        unNormSkeleton=self.get_unnorm(normSkeleton)
        heatmap_to_use = self.pose_to_heatmap(unNormSkeleton, imgSize, 64)

        imageSequence, imagesize = self.get_rgb(view, name_sample)
        # imageSequence, imagesize = self.get_grayscale(view, name_sample)


        dicts={
            'imgSequence_to_use': imageSequence,
            'sequence_to_use': normSkeleton,
            'heatmap_to_use': heatmap_to_use,
            'actLabel':label_action, 'nframes':self.T, 'info':info
        }
        
        return dicts
        

if __name__ == "__main__":
    DS = NUCLAsubject(phase='train')
    dataloader= tudata.DataLoader(DS, batch_size=1, shuffle=False,
                                    num_workers=0,pin_memory=False)
    print(len(DS))
    for i, sample in enumerate(dataloader):
        print(i)
        pass
    pass
