#10/27/2020, Dan
#This file is used as the dataloader to test the cross-view dynamics

import os
import math
import numpy as np
import pickle
import random
import torch
import torch.utils.data as data
from torchvision import transforms
from utils import Gaussian, DrawGaussian


class NTURGBDskeleton(data.Dataset):
    """NTU-RGB+D Skeleton Dataset
    to access input sequence, GT label
    """

    def __init__(self, root_skeleton="/data/Yuexi/NTU-RGBD/skeletons/npy",
                root_list="/data/Yuexi/NTU-RGBD/list/",
                phase='train', cam='1', T=20):
        self.root_skeleton = root_skeleton
        self.root_list = root_list
        self.cam = []
        for name_cam in cam.split(','):
            self.cam.append("C00"+name_cam)
        self.T = T

        self.num_actions = 60
        # Generate the list of files according to cam and phase
        if phase == 'test':
            file_list = os.path.join(self.root_list,f"skeleton_all_{self.cam[0]}_T{self.T}.list")
            self.list_samples = np.loadtxt(file_list, dtype=str)
        else:
            for i, name_cam in enumerate(self.cam):
                file_list = os.path.join(self.root_list,f"skeleton_{phase}_{name_cam}_T{self.T}.list")
                if i == 0:
                    self.list_samples = np.loadtxt(file_list, dtype=str)
                else:
                    self.list_samples = np.concatenate((self.list_samples, np.loadtxt(file_list, dtype=str)), axis=0)
        
        # Compute the MEAN and STD of the dataset
        allSkeleton = []
        for name_sample in self.list_samples:
            skeleton = np.load(os.path.join(self.root_skeleton, name_sample[0]+'.skeleton.npy'), allow_pickle=True).item()['rgb_body0']
            allSkeleton.append(skeleton)
        allSkeleton=np.nan_to_num(np.concatenate(allSkeleton, 0))
        self.x_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:,0], 0)  # 1 x num_joint
        self.y_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:,1], 0)
        self.x_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:,0], 0)
        self.y_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:,1], 0)
    
        # Print
        if phase == 'train':
            self.list_samples = self.list_samples[0:1000]
        else:
            self.list_samples = self.list_samples[0:100]


        print(f"=== {phase} {self.cam}: {len(self.list_samples)} samples ===")
    
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

        # T = poses.shape[0]
        H = image_size[0]
        W = image_size[1]
        heatmaps = []
        for t in range(0, self.T):
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
    """""
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
                transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

            img_tensor = preprocess(input_image)

            imgSequence.append(img_tensor.unsqueeze(0))

        imgSequence = torch.cat((imgSequence), 0)

        return imgSequence, imgSize
    """

    def __getitem__(self, index):
        """
        Return:
            skeletons: FloatTensor, [T, num_joints, 2]
            label_action: int, label for the action
            info: dict['sample_name', 'T_sample', 'time_offset']
        """
        name_sample = self.list_samples[index][0]
        label_action = int(name_sample[-2:])-1 # 0-59
        # print(index, name_sample)
        # skeleton shape (T_sample, 25, 2)
        # scale of 2d skeleton, resolution 1920 x 1080
        skeleton=np.load(os.path.join(self.root_skeleton, name_sample+'.skeleton.npy'), allow_pickle=True).item()['rgb_body0']
        T_sample, num_joints, dim = skeleton.shape

        stride=T_sample/self.T
        ids_sample = []
        for i in range(self.T):
            # print(int(stride*i),int(stride*(i+1))-1)
            # Make sure NAN not in joints data
            while(True):
                val_data = True
                id_sample=random.randint(int(stride*i),int(stride*(i+1))-1)
                # The preprocessing part can make sure there is a valid sequence
                # But this part cannot be deleted
                for data_joint in skeleton[id_sample]:
                    if math.isnan(data_joint[0]) or math.isnan(data_joint[1]):
                        val_data = False
                        break
                if val_data:
                    ids_sample.append(id_sample)
                    break
        
        skeleton_input = skeleton[ids_sample,:,:]
        info = {'name_sample':name_sample, 'T_sample':T_sample, 'time_offset':ids_sample}

        # Prepare Zero Images
        imgSequence = np.zeros((self.T, 3, 224, 224))
        imgSize = (1980, 1080)
        # Prepare Skeleton
        normSkeleton=self.get_norm(skeleton_input)
        # Prepare heatmaps
        unNormSkeleton=self.get_unnorm(normSkeleton)
        heatmap_to_use = self.pose_to_heatmap(unNormSkeleton, imgSize, 64)




        dicts={
            'imgSequence_to_use': imgSequence,
            'normSkeleton': normSkeleton, 'unNormSkeleton':unNormSkeleton,
            'heatmap_to_use': heatmap_to_use,
            'actLabel':label_action, 'nframes':self.T, 'info':info
        }
        
        return dicts

if __name__ == "__main__":
    DS=NTURGBDskeleton(root_list=f"/data/Dan/NTU-RGBD/list/fast_25/T20",cam='3', phase='test')
    dataloader = torch.utils.data.DataLoader(DS, batch_size=1, shuffle=False,
                                            num_workers=1, pin_memory=True)
    print(len(DS))

    for i, sample in enumerate(dataloader):

        print(i)
        pass
    pass