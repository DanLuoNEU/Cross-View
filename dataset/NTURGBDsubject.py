
#This file is used as the dataloader to test the cross-view dynamics

import os
import math
import numpy as np
import pickle
import random
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import json

from utils import Gaussian, DrawGaussian
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

class NTURGBDsubject(data.Dataset):
    """NTU-RGB+D Skeleton Dataset - Cross-Subject
    to access input sequence, GT label
    """
    def __init__(self, root_skeleton, nanList, dataType,clip, phase, T):
        self.root_skeleton = root_skeleton
        self.root_image = '/data/NTU-RGBD/frames'
        # self.root_list = root_list
        self.T = T
        self.phase = phase
        self.nanList = nanList
        self.dataType = dataType
        self.num_actions = 60
        self.clip = clip
        self.clips = 8
        self.trainID = ['P001', 'P002', 'P004', 'P005','P008', 'P009', 'P013', 'P014', 'P015', 'P016', 'P017', 'P018',
                        'P019', 'P025', 'P027', 'P028', 'P031', 'P034', 'P035', 'P038']
        self.testID = ['P003', 'P007', 'P010', 'P011', 'P012', 'P020', 'P021', 'P022', 'P023', 'P024', 'P026','P029',
                       'P030', 'P032', 'P033', 'P036', 'P037', 'P039', 'P040']



        # Generate the list of files according to phase
        # if phase == 'test':
        #     self.file_list = os.path.join(self.root_list,f"skeleton_test_subj_T{self.T}.list")
        # else:
        #     self.file_list = os.path.join(self.root_list,f"skeleton_train_subj_T{self.T}_0.1.list")
        # self.list_samples = np.loadtxt(self.file_list, dtype=str)
        file_list = os.listdir(self.root_skeleton)
        self.train_list = []
        self.test_list = []

        for file_name in file_list:
            if file_name not in self.nanList:
                Pid = file_name.split('.')[0].split('R')[0][-4:]

                if Pid in self.trainID:
                    self.train_list.append(file_name.split('.')[0])
                else:
                    self.test_list.append(file_name.split('.')[0])

        self.random_list = self.train_list[0:500]
        # Compute the MEAN and STD of the dataset
        allSkeleton = []
        for name_sample in self.random_list:
            if self.dataType == '2D':
                # skeleton1 = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()[
                #     'rgb_body0']
                skeleton1, _ = getJsonData(self.root_skeleton, name_sample)
                # for t in range(0, skeleton1.shape[0]):
                #     for joint in skeleton1[t]:
                #         if np.isnan(joint[0]) == True or np.isnan(joint[1]) == True:
                # print('name_sample:', name_sample, 'joint:', joint)
                # print('check')

            else:
                skeleton1 = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()[
                    'skel_body0']
            allSkeleton.append(skeleton1)


        # allSkeleton=np.nan_to_num(np.concatenate(allSkeleton, 0))# Transfer nan to zero!!!
        allSkeleton = np.concatenate((allSkeleton), 0)
        if self.dataType == '2D':
            self.x_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 0], 0)  # 1 x num_joint
            self.y_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 1], 0)
            self.x_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 0], 0)
            self.y_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 1], 0)
        else:
            self.x_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 0], 0)  # 1 x num_joint
            self.y_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 1], 0)
            self.x_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 0], 0)
            self.y_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 1], 0)
            self.z_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 2], 0)
            self.z_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 2], 0)

        if self.phase == 'train':
            self.samples_list = self.random_list
        else:
            # self.samples_list = random.sample(self.test_list,50)
            self.samples_list = random.sample(self.train_list[400:550], 30)
            # self.samples_list = self.random_list

            """""
            self.test_names = []
            for item in self.random_list:

                # string = item.split(self.target_view)
                # name = ''.join((string[0], self.test_view, string[1]))
                Pid = item.split('.')[0].split('R')[0][-4:]
                string = item.split(Pid)

                for id in self.testID:
                    name = string[0] + id + string[1]
                    if name in self.test_list:
                        self.test_names.append(name)
            
            self.samples_list = random.sample(self.test_names, 20)
            """

        print(f"=== {phase} Cross-Subject: {len(self.samples_list)} samples ===")
    
    def __len__(self):
        return len(self.samples_list)
        # return 1

    def get_norm(self, skeleton):
        '''Get the normalization according to the skeleton
        Arguments:
            skeleton: T(8) x num_joints(25) x 2, unnormalized in original resolution
        Return:
            skeleton: T(8) x num_joints(25) x 2, normalized in original resolution
        '''
        '2D data'
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
            Y = normSkeleton[:, :, 1]  # inputLen x 15
            unNormX = X * stdX_mat + meanX_mat
            unNormY = Y * stdY_mat + meanY_mat
            unNormSkeleton = torch.cat((unNormX.unsqueeze(2), unNormY.unsqueeze(2)), 2)
        else:
            X = normSkeleton[:, :, 0]
            Y = normSkeleton[:, :, 1]
            Z = normSkeleton[:, :, 2]
            meanZ_mat = torch.FloatTensor(self.z_mean_skeleton).repeat(framNum, 1)
            stdZ_mat = torch.FloatTensor(self.z_std_skeleton).repeat(framNum, 1)
            unNormZ = Z * stdZ_mat + meanZ_mat
            unNormSkeleton = torch.cat((X.unsqueeze(2), Y.unsqueeze(2), unNormZ.unsqueeze(2)), 2)

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

    def get_rgb(self, name_sample):
        data_path = os.path.join(self.root_image, name_sample)
        # print(data_path)
        # fileList = np.loadtxt(os.path.join(data_path, 'fileList.txt'))
        imgId = []
        # for l in fileList:
        #     imgId.append(int(l[0]))
        # imgId.sort()

        imageList = []

        for item in os.listdir(data_path):
            if item.find('.jpg') != -1:
                id = int(item.split('_')[1].split('.jpg')[0])
                imgId.append(id)

        imgId.sort()

        for i in range(0, len(imgId)):
            for item in os.listdir(data_path):
                if item.find('.jpg') != -1:
                    if int(item.split('_')[1].split('.jpg')[0]) == imgId[i]:
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
    def get_data(self, name_sample):
        # skeleton = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()[
        #     'rgb_body0']
        imageSequence, _ = self.get_rgb(name_sample)
        if self.dataType == '2D':

            # skeleton = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()[
            #         'rgb_body0']
            skeleton, usedID = getJsonData(self.root_skeleton, name_sample)
            imageSequence = imageSequence[usedID]
        else:
            skeleton = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()['skel_body0']

        T_sample, num_joints, dim = skeleton.shape


        if T_sample <= self.T:
            skeleton_input = skeleton
            imageSequence_input = imageSequence
            T_sample = self.T
            ids_sample = np.linspace(0, T_sample - 1, T_sample, dtype=np.int)
        else:

            # stride = T_sample / self.T
            # ids_sample = []
            # for i in range(self.T):
            #     id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
            #     ids_sample.append(id_sample)
            #
            # skeleton_input = skeleton[ids_sample, :, :]
            # imageSequence_input = imageSequence[ids_sample]


            stride = T_sample / self.T
            ids_sample = []
            for i in range(self.T):
                # print(int(stride*i),int(stride*(i+1))-1)
                # Make sure NAN not in joints data
                while (True):
                    val_data = True
                    id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
                    # The preprocessing part can make sure there is a valid sequence
                    # But this part cannot be deleted
                    for data_joint in skeleton[id_sample]:
                        if math.isnan(data_joint[0]) or math.isnan(data_joint[1]):
                            val_data = False
                            break
                    if val_data:
                        ids_sample.append(id_sample)
                        break

            skeleton_input = skeleton[ids_sample, :, :]
            imageSequence_input = imageSequence[ids_sample]
        info = {'name_sample': name_sample, 'T_sample': T_sample, 'time_offset': ids_sample}

        # Prepare Zero Images
        # imgSequence = np.zeros((self.T, 3, 224, 224))
        imgSize = (1980, 1080)
        # Prepare Skeleton
        normSkeleton = self.get_norm(skeleton_input)
        # Prepare heatmaps
        unNormSkeleton = self.get_unnorm(normSkeleton)
        # heatmap_to_use = self.pose_to_heatmap(unNormSkeleton, imgSize, 64)

        skeletonData = {'normSkeleton': normSkeleton, 'unNormSkeleton': unNormSkeleton}
        # return heatmap_to_use, skeletonData, imageSequence_input, info
        return skeletonData, imageSequence_input, info

    def get_data_multi(self, name_sample):
        imageSequence, _ = self.get_rgb(name_sample)
        if self.dataType == '2D':
            # skeleton = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()[
            #     'rgb_body0']
            skeleton, usedID = getJsonData(self.root_skeleton, name_sample)
            imageSequence = imageSequence[usedID]
            normSkeleton = self.get_norm(skeleton)
            # T_sample, num_joints, dim = normSkeleton.shape
        # else:
        #     skeleton = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()[
        #         'skel_body0']

            T_sample, num_joints, dim = skeleton.shape
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
            L = 36 # clip length
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

                    if id - int(L/2) < 0 and T_sample - (id + int(L/2)) >= 0:
                        # temp1 = skeleton[0:id]
                        # temp2 = skeleton[id:]
                        temp = np.expand_dims(normSkeleton[0:L], 0)
                        tempImg = np.expand_dims(imageSequence[0:L], 0)
                    elif id - int(L/2) >= 0 and T_sample - (id+int(L/2)) >= 0:
                        temp = np.expand_dims(normSkeleton[id-int(L/2):id+int(L/2)], 0)
                        tempImg = np.expand_dims(imageSequence[id-int(L/2):id+int(L/2)], 0)

                    elif id - int(L/2) >= 0 and T_sample - (id+int(L/2)) < 0:
                        temp = np.expand_dims(normSkeleton[T_sample-L:], 0)
                        tempImg = np.expand_dims(imageSequence[T_sample-L:], 0)



                    allSkeletons.append(temp)
                    allImageSequence.append(tempImg)

                allSkeletons = np.concatenate((allSkeletons), 0)
                allImageSequence = np.concatenate((allImageSequence), 0)
        else:
            skeleton = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()['skel_body0']
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
        info = {'name_sample': name_sample, 'T_sample': T_sample}

        return allSkeletons, allImageSequence, info

    def __getitem__(self, index):
        """
        Return:
            skeletons: FloatTensor, [T, num_joints, 2]
            label_action: int, label for the action
            info: dict['sample_name', 'T_sample', 'time_offset']
        """
        name_sample = self.samples_list[index]
        label_action = int(name_sample[-2:])-1 # 0-59


        if self.clip == 'Single':
            input_skeleton, input_image, input_info = self.get_data(name_sample)
            dicts = {'input_image': input_image,
                     'input_skeleton': input_skeleton,
                     'input_info': input_info, 'action': label_action}
        else:

            input_multiSkeletons, input_multiImage,info = self.get_data_multi(name_sample)
            # testVelocity = self.getVelocity(testView_multiSkeletons)
            dicts = {'input_skeleton': input_multiSkeletons, 'input_image': input_multiImage,
                     'action': label_action, 'info':info}


        return dicts

if __name__ == "__main__":
    # root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    dataType = '2D'
    if dataType == '3D':
        root_skeleton = "/data/Yuexi/NTU-RGBD/skeletons/npy"
    else:
        root_skeleton = "/data/NTU-RGBD/poses"
    nanList = list(np.load('../NTU_badList.npz')['x'])

    DS=NTURGBDsubject(root_skeleton, nanList, dataType='2D',clip='Multi', phase='test', T=36)
    dataloader = torch.utils.data.DataLoader(DS, batch_size=1, shuffle=False,
                                            num_workers=4, pin_memory=True)
    print(len(DS))

    for i, (sample) in enumerate(dataloader):
        print('sample:', i)

        # input_skeleton = sample['input_skeleton']['normSkeleton'].squeeze(0)
        input_skeleton = sample['input_skeleton'].squeeze(0)

        input_image = sample['input_image']
        for clip in range(0, 10):
            for t in range(0, input_skeleton[clip].shape[0]):
                for joint in input_skeleton[clip, t]:
                    if torch.isnan(joint[0]) == True or torch.isnan(joint[1]) == True :
                        print('nan:', sample['input_info']['name_sample'][0])
                    elif joint[0] == 0 and joint[1] == 0:
                        print('zero joint:',sample['input_info']['name_sample'][0])


    print('done')