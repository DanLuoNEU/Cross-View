import os
import math
import numpy as np
import pickle
import random
import torch
import torch.utils.data as data
from torchvision import transforms
from utils import Gaussian, DrawGaussian
from PIL import Image
import json

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

class NTURGBD_viewProjection(data.Dataset):
    """NTU-RGB+D Skeleton Dataset
    to access input sequence, GT label
    """

    def __init__(self, root_skeleton, root_list, nanList, dataType,clip, phase, T, target_view, project_view, test_view):
        self.root_skeleton = root_skeleton
        self.root_list = root_list
        self.root_image = '/data/NTU-RGBD/frames'
        self.T = T
        self.target_view = target_view
        self.project_view = project_view
        self.test_view = test_view
        self.phase = phase
        self.dataType = dataType
        self.nanList = nanList
        self.clips = 10
        self.clip = clip

        target_file = os.path.join(self.root_list, f"skeleton_all_{self.target_view}.list")
        project_file = os.path.join(self.root_list, f"skeleton_all_{self.project_view}.list")

        self.target_list = []
        target_txt = np.loadtxt(target_file, dtype=str)
        for item in target_txt:
            if item[0] not in self.nanList:
                self.target_list.append(item[0])

        self.project_list = []
        project_txt = np.loadtxt(project_file, dtype=str)
        for item in project_txt:
            if item[0]  not in self.nanList:
                self.project_list.append(item[0])

        self.list_to_use = []

        for item in self.project_list:
            string = item.split(self.project_view)
            name = ''.join((string[0],self.target_view, string[1]))

            if name in self.target_list:
                self.list_to_use.append(name)    # based on target_file

        # L = len(self.list_to_use)

        # self.list_name = self.list_to_use[0:int(0.9*L)]
            # self.list_train = random.sample(self.list_name, int(0.1*len(self.list_name)))
        # # Compute the MEAN and STD of the dataset
        allSkeleton = []
        # self.samples_list = random.sample(self.list_to_use, 2000)
        self.samples_list = self.list_to_use

        
        for name_sample in self.samples_list:

            # skeleton = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()['skel_body0'] # 3D, if ['rgb_body0'],2D
            #
            # allSkeleton.append(skeleton)

            temp = name_sample.split(self.target_view)
            rename = ''.join((temp[0], self.project_view, temp[1]))
            # skeleton = np.load(os.path.join(self.root_skeleton, rename + '.skeleton.npy'), allow_pickle=True).item()['skel_body0']
            # if name_sample + '.skeleton.npy' not in self.nanList:
            if name_sample not in self.nanList:
                if self.dataType == '2D':
                    # print('sample_name:', name_sample)
                    # skeleton1 = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()['rgb_body0']
                    skeleton1, _ = getJsonData(self.root_skeleton, name_sample)
                else:
                    skeleton1 = np.load(os.path.join(self.root_skeleton, name_sample + '.skeleton.npy'), allow_pickle=True).item()['skel_body0']
                allSkeleton.append(skeleton1)

            # elif rename + '.skeleton.npy' not in self.nanList:
            elif rename not in self.nanList:
                if self.dataType == '2D':
                    # skeleton2 = np.load(os.path.join(self.root_skeleton, rename + '.skeleton.npy'), allow_pickle=True).item()['rgb_body0']
                    skeleton2, _ = getJsonData(self.root_skeleton, name_sample)
                    # for t in range(0, skeleton2.shape[0]):
                    #     for joint in skeleton2[t]:
                    #         if np.isnan(joint[0]) == True or np.isnan(joint[1]) == True:
                    #             print('name_sample:', rename, 'joint:', joint)
                    #             print('check')
                else:
                    skeleton2 = np.load(os.path.join(self.root_skeleton, rename + '.skeleton.npy'), allow_pickle=True).item()['skel_body0']
                allSkeleton.append(skeleton2)
            else:

                # print('target name:', name_sample, 'project name:', rename)
                continue
            # allSkeleton.append(skeleton)

        # allSkeleton = np.nan_to_num(np.concatenate(allSkeleton, 0))
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
            # self.samples_list = random.sample(self.list_to_use, 500)
            self.samples_list = self.samples_list[0:1000]

        else:
            test_file = os.path.join(self.root_list, f"skeleton_all_{self.test_view}.list")
            self.test_list = []
            test_txt = np.loadtxt(test_file, dtype=str)

            for item in test_txt:
                if item[0]  not in self.nanList:
                    self.test_list.append(item[0])
                else:
                    continue

            self.test_names = []
            for item in self.list_to_use:
                string = item.split(self.target_view)
                name = ''.join((string[0], self.test_view, string[1]))

                if name in test_txt:
                    self.test_names.append(name)

            # self.samples_list = random.sample(self.test_names, 100)
            self.samples_list = self.test_list[0:100]



        # Print
        print(f"=== {self.phase}: {len(self.samples_list)} samples ===")

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
        # name_sample_target = self.samples_list[index]
        # string = name_sample_target.split(self.target_view)
        # name_sample_project = ''.join((string[0], self.project_view, string[1]))

        name_sample = self.samples_list[index]
        label_action = int(name_sample[-2:]) - 1  # 0-59


        # action_name = name_sample_[-4:]
        if self.phase == 'test':
            # list = [name_sample_target, name_sample_project]
            # name_sample = random.choice(list)

            if self.clip == 'Single':
                test_view_skeleton, test_view_image, test_info = self.get_data(name_sample)
                dicts = {'input_image': test_view_image,
                         'input_skeleton': test_view_skeleton,
                         'input_info': test_info, 'action': label_action}
            else:

                testView_multiSkeletons, testView_multiImage,info = self.get_data_multi(name_sample)
                # testVelocity = self.getVelocity(testView_multiSkeletons)
                dicts = {'input_skeleton': testView_multiSkeletons, 'input_image': testView_multiImage,
                         'action': label_action, 'info':info}

        else:
            name_sample_target = self.samples_list[index]
            string = name_sample_target.split(self.target_view)
            name_sample_project = ''.join((string[0], self.project_view, string[1]))
            if self.clip == 'Single':
                target_view_skeleton, target_view_image, target_info = self.get_data(name_sample_target)
                projet_view_skeleton, project_view_image, project_info = self.get_data(name_sample_project)
                dicts = { 'target_image': target_view_image,
                         'target_skeleton': target_view_skeleton, 'target_info': target_info,
                          'project_image': project_view_image,
                         'project_skeleton': projet_view_skeleton, 'project_info': project_info, 'action': label_action}
            else:
                projectView_multiSkeleton, projectView_multiImage, info_target = self.get_data_multi(name_sample_target)
                targetView_multiSkeleton, targetView_multiImage, info_project = self.get_data_multi(name_sample_project)

                dicts = {'target_skeleton': targetView_multiSkeleton, 'target_image': targetView_multiImage,
                         'project_skeleton': projectView_multiSkeleton, 'project_image': projectView_multiImage,
                         'action': label_action, 'target_info':info_target, 'project_info':info_project}

        return dicts


if __name__ == "__main__":
    # nanList = list(np.load('../NTU_badList.npz')['x'])
    with open('/data/NTU-RGBD/ntu_rgb_missings_60.txt', 'r') as f:
        nanList = f.readlines()
        nanList = [line.rstrip() for line in nanList]
    DS = NTURGBD_viewProjection(root_skeleton="/data/NTU-RGBD/poses_60",
                 root_list="/data/NTU-RGBD/list/",nanList=nanList, dataType='2D', clip='Multi',
                 phase='test', T=36,target_view='C002', project_view='C003', test_view='C001')

    dataloader = torch.utils.data.DataLoader(DS, batch_size=1, shuffle=False,
                                             num_workers=4, pin_memory=True)
    print(len(DS))

    """""
    for i, sample in enumerate(dataloader):
        print('sample:', i)
        # input_skeleton = sample['test_skeleton'].squeeze(0)
        # input_image = sample['test_image']
        input_skeleton_target = sample['target_skeleton'].squeeze(0)
        input_skeleton_project = sample['project_skeleton'].squeeze(0)
        for clip in range(0, 6):

            for t in range(0, input_skeleton_target[clip].shape[0]):
                for joint in input_skeleton_target[clip,t]:
                    if torch.isnan(joint[0]) == True or torch.isnan(joint[1]) == True:
                        print('nan:', sample['target_info']['name_sample'][0])
                    elif joint[0] == 0 and joint[1] == 0:
                        print('zero joint:',sample['target_info']['name_sample'][0])

            for t in range(0, input_skeleton_project[clip].shape[0]):
                for joint in input_skeleton_project[clip, t]:
                    if torch.isnan(joint[0]) == True or torch.isnan(joint[1]) == True:
                        print('nan:',sample['project_info']['name_sample'][0])
                    elif joint[0] == 0 and joint[1] == 0:
                        print('zero joint:',sample['project_info']['name_sample'][0])
        # print('sample:',i, 'skeleton:', input_skeleton.shape, 'image:', input_image.shape)
        print('sample:', i, 'target:', input_skeleton_target.shape, 'project:', input_skeleton_project.shape)
    """
    print('done')
