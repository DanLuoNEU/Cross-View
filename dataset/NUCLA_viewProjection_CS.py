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
import json
import cv2
"""
   Hip_Center = 1;   Spine = 2;       Shoulder_Center = 3; Head = 4;           Shoulder_Left = 5;
   Elbow_Left = 6;   Wrist_Left = 7;  Hand_Left = 8;       Shoulder_Right = 9; Elbow_Right = 10;
   Wrist_Right = 11; Hand_Right = 12; Hip_Left = 13;       Knee_Left = 14;     Ankle_Left = 15;
   Foot_Left = 16;   Hip_Right = 17;  Knee_Right = 18;     Ankle_Right = 19;   Foot_Right = 20;
"""


def getJsonData(fileRoot, folder):
    skeleton = []
    allFiles = os.listdir(os.path.join(fileRoot, folder))
    allFiles.sort()
    usedID = []
    for i in range(0, len(allFiles)):
        with open(os.path.join(fileRoot,folder, allFiles[i])) as f:
            data = json.load(f)
        # print(len(data['people']))
        if len(data['people']) != 0:
            # print('check')
            usedID.append(i)
            temp = data['people'][0]['pose_keypoints_2d']
            pose = np.expand_dims(np.asarray(temp).reshape(25, 3)[:,0:2], 0)
            skeleton.append(pose)
        else:
            continue

    skeleton = np.concatenate((skeleton))
    return skeleton, usedID

class NUCLAsubject(tudata.Dataset):
    """Northeastern-UCLA Dataset Skeleton Dataset
        Access input skeleton sequence, GT label
        When T=0, it returns the whole 
    """
    def __init__(self, root_list, dataType, clip, phase, T):
        # self.root_skeleton = root_skeleton
        self.data_root = '/data/N-UCLA_MA_3D/multiview_action'
        self.dataType = dataType
        if self.dataType == '2D':
            self.root_skeleton = '/data/N-UCLA_MA_3D/openpose_est'
        else:
            self.root_skeleton = '/data/N-UCLA_MA_3D/skeletons_3d'
        self.root_list = root_list
        self.dataType = dataType
        self.clip = clip  # Single/Multi
        self.T = T
        self.num_samples = 0
        self.clips = 6
        self.phase = phase

        self.num_action = 10
        self.action_list={'a01':0, 'a02':1, 'a03':2, 'a04':3, 'a05':4,
                        'a06':5, 'a08':6, 'a09':7, 'a11':8, 'a12':9}
        self.actions={'a01':'pick up with one hand', 'a02':"pick up with two hands", 'a03':"drop trash", 'a04':"walk around", 'a05':"sit down",
                        'a06':"stand up", 'a08':"donning", 'a09':"doffing", 'a11':"throw", 'a12':"carry"}

        # Get the list of files according to cam and phase
        self.samples_list = []
        # Compute the MEAN and STD of the dataset
        allSkeleton = []
        file_list = os.path.join(self.root_list, f"subject_{self.phase}.list")
        list_samples = np.loadtxt(file_list, dtype=str)
        for sample in list_samples:
            view, name_sample = sample.split(',')[0], sample.split(',')[1]
            self.samples_list.append((view, name_sample))
            if self.dataType == '2D':
                skeleton, _ = getJsonData(os.path.join(self.root_skeleton, view), name_sample)
            else:
                skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)

            if len(skeleton.shape):
                np.asarray(skeleton)
            allSkeleton.append(skeleton)
        allSkeleton=np.concatenate(allSkeleton, 0)#((allSkeleton),0)
        if self.dataType == '2D':
            self.x_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 0], 0)  # 1 x num_joint
            self.y_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 1], 0)
            self.x_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 0], 0)
            self.y_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 1], 0)
        else:
            self.x_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 0], 0)  # 1 x num_joint
            self.y_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 1], 0)
            self.z_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 2], 0)

            self.x_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 0], 0)
            self.y_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 1], 0)
            self.z_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 2], 0)
        # Print

        if self.phase == 'test':
            # self.samples_list = random.sample(self.samples_list,100)
            self.samples_list = self.samples_list
        print(f"=== {phase} subject: {len(self.samples_list)} samples ===")

    def __len__(self):
        return len(self.samples_list)
        # return 1  # debug

    def get_norm(self, skeleton):
        '''Get the normalization according to the skeleton
        Arguments:
            skeleton: T(8) x num_joints(25) x 2, unnormalized in original resolution
        Return:
            skeleton: T(8) x num_joints(25) x 2, normalized in original resolution
        '''
        if self.dataType == '2D':
            X = skeleton[:, :, 0]
            Y = skeleton[:, :, 1]
            normX = (X - self.x_mean_skeleton) / self.x_std_skeleton
            normY = (Y - self.y_mean_skeleton) / self.y_std_skeleton
            normSkeleton = np.concatenate((np.expand_dims(normX, 2), np.expand_dims(normY, 2)), 2).astype(float)


        else:
            X = skeleton[:, :, 0]
            Y = skeleton[:, :, 1]
            Z = skeleton[:, :, 2]

            normX = (X - self.x_mean_skeleton) / self.x_std_skeleton
            normY = (Y - self.y_mean_skeleton) / self.y_std_skeleton
            normZ = (Z - self.z_mean_skeleton) / self.z_std_skeleton
            normSkeleton = np.concatenate(
                (np.expand_dims(normX, 2), np.expand_dims(normY, 2), np.expand_dims(normZ, 2)), 2).astype(float)

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
        if self.dataType == '2D':
            meanX_mat = torch.FloatTensor(self.x_mean_skeleton).repeat(framNum, 1)  # inputLen x 15
            meanY_mat = torch.FloatTensor(self.y_mean_skeleton).repeat(framNum, 1)
            stdX_mat = torch.FloatTensor(self.x_std_skeleton).repeat(framNum, 1)
            stdY_mat = torch.FloatTensor(self.y_std_skeleton).repeat(framNum, 1)
            X = normSkeleton[:, :, 0]  # inputLen x 15
            Y = normSkeleton[:, :, 1]
            unNormX = X * stdX_mat + meanX_mat
            unNormY = Y * stdY_mat + meanY_mat
            unNormSkeleton = torch.cat((unNormX.unsqueeze(2), unNormY.unsqueeze(2)), 2)

        else:

            meanX_mat = torch.FloatTensor(self.x_mean_skeleton).repeat(framNum, 1)  # inputLen x 15
            meanY_mat = torch.FloatTensor(self.y_mean_skeleton).repeat(framNum, 1)
            meanZ_mat = torch.FloatTensor(self.z_mean_skeleton).repeat(framNum, 1)

            stdX_mat = torch.FloatTensor(self.x_std_skeleton).repeat(framNum, 1)
            stdY_mat = torch.FloatTensor(self.y_std_skeleton).repeat(framNum, 1)
            stdZ_mat = torch.FloatTensor(self.z_std_skeleton).repeat(framNum, 1)

            X = normSkeleton[:, :, 0]  # inputLen x 15
            Y = normSkeleton[:, :, 1]  # inputLen x 15
            Z = normSkeleton[:, :, 2]

            unNormX = X * stdX_mat + meanX_mat
            unNormY = Y * stdY_mat + meanY_mat
            unNormZ = Z * stdZ_mat + meanZ_mat

            # unNormSkeleton = torch.cat((unNormX.unsqueeze(2), unNormY.unsqueeze(2)), 2)
            unNormSkeleton = torch.cat((unNormX.unsqueeze(2), unNormY.unsqueeze(2), unNormZ.unsqueeze(2)), 2)
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
        imgSequenceOrig = []

        for i in range(0, len(imageList)):
            img_path = os.path.join(data_path, imageList[i])
            orig_image = cv2.imread(img_path)
            imgSequenceOrig.append(np.expand_dims(orig_image,0))

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
        imgSequenceOrig = np.concatenate((imgSequenceOrig), 0)

        return imgSequence, imgSize, imgSequenceOrig


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

    def get_data(self, view, name_sample):

        imageSequence, _, imageSequence_orig = self.get_rgb(view, name_sample)
        if self.dataType == '2D':
            skeleton, usedID = getJsonData(os.path.join(self.root_skeleton, view), name_sample)
            imageSequence = imageSequence[usedID]
        else:
            skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)
        #
        T_sample, num_joints, dim = skeleton.shape
        #

        # if imageSequence_orig.shape[0] != T_sample:
        #     print('check')

        if self.T == 0:
            skeleton_input = skeleton
            imageSequence_input = imageSequence
            # imgSequence = np.zeros((T_sample, 3, 224, 224))
            details = {'name_sample': name_sample, 'T_sample': T_sample, 'time_offset': range(T_sample), 'view':view}
        else:
            if T_sample < self.T:
                """""
                Tadd = abs(T_sample - self.T)
                last = np.expand_dims(skeleton[-1,:,:],0)
                copyLast = np.repeat(last, Tadd, 0)
                skeleton = np.concatenate((skeleton, copyLast), 0) # copy last frame Tadd times

                lastImg = np.expand_dims(imageSequence_orig[-1,:,:,:], 0)
                copyLastImg = np.repeat(lastImg, Tadd, 0)
                imageSequence_orig = np.concatenate((imageSequence_orig, copyLastImg),0)
                """
                skeleton_input = skeleton
                imageSequence_input = imageSequence
                T_sample = self.T
                ids_sample = np.linspace(0, T_sample-1, T_sample, dtype=np.int)

            else:

                stride = T_sample / self.T
                ids_sample = []
                for i in range(self.T):
                    id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
                    ids_sample.append(id_sample)

                skeleton_input = skeleton[ids_sample, :, :]
                imageSequence_input = imageSequence[ids_sample]


            details = {'name_sample': name_sample, 'T_sample': T_sample, 'time_offset': ids_sample,'view':view}

            # Prepare Zero Images
            # imgSequence = np.zeros((self.T, 3, 224, 224))
        imgSize = (640, 480)
        # Prepare Skeleton
        normSkeleton = self.get_norm(skeleton_input)
        # Prepare heatmaps
        unNormSkeleton = self.get_unnorm(normSkeleton)
        heatmap_to_use = self.pose_to_heatmap(unNormSkeleton, imgSize, 64)
        # imageSequence, _ = self.get_rgb(view, name_sample)
        # imageSequence, _ = self.get_grayscale(view, name_sample)

        skeletonData = {'normSkeleton':normSkeleton, 'unNormSkeleton':unNormSkeleton}
        return heatmap_to_use, imageSequence_input, skeletonData, details

    def get_data_multi(self, view, name_sample):
        imageSequence, _, imageSequence_orig = self.get_rgb(view, name_sample)
        if self.dataType == '2D':
            skeleton, usedID = getJsonData(os.path.join(self.root_skeleton, view), name_sample)
            imageSequence = imageSequence[usedID]
            normSkeleton = self.get_norm(skeleton)
            T_sample, num_joints, dim = normSkeleton.shape

            if T_sample < self.clips:
                Tadd = abs(T_sample - self.clips)
                last = np.expand_dims(normSkeleton[-1, :, :], 0)
                copyLast = np.repeat(last, Tadd, 0)
                normSkeleton = np.concatenate((normSkeleton, copyLast), 0)  # copy last frame Tadd times

                lastImg = np.expand_dims(imageSequence[-1, :, :, :], 0)
                copyLastImg = np.repeat(lastImg, Tadd, 0)
                imageSequence = np.concatenate((imageSequence, copyLastImg), 0)

                T_sample = self.clips

            stride = T_sample / self.clips
            ids_sample = []
            for i in range(self.clips):
                id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
                ids_sample.append(id_sample)
            L = 36  # clip length
            if T_sample <= L:
                """""
                Tadd = abs(T_sample - L)
                last = np.expand_dims(normSkeleton[-1, :, :], 0)
                copyLast = np.repeat(last, Tadd, 0)
                temp = np.expand_dims(np.concatenate((normSkeleton, copyLast), 0), 0)
                allSkeletons = np.repeat(temp, self.clips, 0)
                """
                temp = np.expand_dims(normSkeleton, 0)
                allSkeletons = np.repeat(temp, self.clips, 0)

                tempImg = np.expand_dims(imageSequence, 0)
                allImageSequence = np.repeat(tempImg, self.clips, 0)


            else:
                allSkeletons = []
                allImageSequence = []
                for id in ids_sample:

                    if id - int(L / 2) < 0 and T_sample - (id + int(L / 2)) >= 0:
                        # temp1 = skeleton[0:id]
                        # temp2 = skeleton[id:]
                        temp = np.expand_dims(normSkeleton[0:L], 0)
                        tempImg = np.expand_dims(imageSequence[0:L], 0)
                    elif id - int(L / 2) >= 0 and T_sample - (id + int(L / 2)) >= 0:
                        temp = np.expand_dims(normSkeleton[id - int(L / 2):id + int(L / 2)], 0)
                        tempImg = np.expand_dims(imageSequence[id - int(L / 2):id + int(L / 2)], 0)

                    elif id - int(L / 2) >= 0 and T_sample - (id + int(L / 2)) < 0:
                        temp = np.expand_dims(normSkeleton[T_sample - L:], 0)
                        tempImg = np.expand_dims(imageSequence[T_sample - L:], 0)

                    allSkeletons.append(temp)
                    allImageSequence.append(tempImg)

                allSkeletons = np.concatenate((allSkeletons), 0)
                allImageSequence = np.concatenate((allImageSequence), 0)
        else:
            skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)
            normSkeleton = self.get_norm(skeleton)
            T_sample, num_joints, dim = normSkeleton.shape

            if T_sample < self.clips:
                Tadd = abs(T_sample - self.clips)
                last = np.expand_dims(normSkeleton[-1, :, :], 0)
                copyLast = np.repeat(last, Tadd, 0)
                normSkeleton = np.concatenate((normSkeleton, copyLast), 0)  # copy last frame Tadd times
                T_sample = self.clips

            stride = T_sample / self.clips
            ids_sample = []
            for i in range(self.clips):
                id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
                ids_sample.append(id_sample)
            L = 36  # clip length
            if T_sample <= L:
                temp = np.expand_dims(normSkeleton, 0)
                allSkeletons = np.repeat(temp, self.clips, 0)
                allImageSequence = np.zeros((self.clips, T_sample, 3, 224, 224))

            else:
                allSkeletons = []
                for id in ids_sample:

                    if id - int(L / 2) < 0 and T_sample - (id + int(L / 2)) >= 0:
                        temp = np.expand_dims(normSkeleton[0:L], 0)

                    elif id - int(L / 2) >= 0 and T_sample - (id + int(L / 2)) >= 0:
                        temp = np.expand_dims(normSkeleton[id - int(L / 2):id + int(L / 2)], 0)


                    elif id - int(L / 2) >= 0 and T_sample - (id + int(L / 2)) < 0:
                        temp = np.expand_dims(normSkeleton[T_sample - L:], 0)

                    allSkeletons.append(temp)

                allSkeletons = np.concatenate((allSkeletons), 0)
                allImageSequence = np.zeros((self.clips, L, 3, 224, 224))

        return allSkeletons, allImageSequence

    def getVelocity(self, skeleton):
        velocity = []
        for i in range(0, skeleton.shape[0]):
            t = skeleton[i].shape[0]
            v = skeleton[i, 1:t] - skeleton[i, 0:t - 1]
            velocity.append(np.expand_dims(v, 0))
        velocity = np.concatenate((velocity), 0)

        return velocity

    def __getitem__(self, index):
        """
        Return:
            skeletons: FloatTensor, [T, num_joints, 2]
            label_action: int, label for the action
            info: dict['sample_name', 'T_sample', 'time_offset']
        """
        view, name_sample = self.samples_list[index]
        label_action = self.action_list[name_sample[:3]]

        if self.clip == 'Single':
            input_heat, input_image, input_skeleton, input_info = self.get_data(view, name_sample)
            dicts = {'input_heat': input_heat, 'input_image': input_image,
                     'input_skeleton': input_skeleton, 'input_info': input_info, 'action': label_action}
        else:
            multiClipSkeleton, multiImage = self.get_data_multi(view, name_sample)
            skeletonVelocity = self.getVelocity(multiClipSkeleton)
            dicts = {'input_skeleton':multiClipSkeleton, 'input_image': multiImage, 'velocity':skeletonVelocity, 'action': label_action}



        return dicts

if __name__ == "__main__":
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    root_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    DS = NUCLAsubject(root_list, dataType='3D', clip='Multi', phase='test', T=36)
    dataloader= tudata.DataLoader(DS, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    print(len(DS))
    for i, sample in enumerate(dataloader):
        print(i, sample['input_skeleton'].shape)

        pass
    print('done')

