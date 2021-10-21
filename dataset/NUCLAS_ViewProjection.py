import os
import math
import numpy as np
import pickle
import random
import torch
import torch.utils.data as tudata
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance

from utils import Gaussian, DrawGaussian

"""
   Hip_Center = 1;
   Spine = 2;
   Shoulder_Center = 3;
   Head = 4;
   Shoulder_Left = 5;
   Elbow_Left = 6;
   Wrist_Left = 7;
   Hand_Left = 8;
   Shoulder_Right = 9;
   Elbow_Right = 10;
   Wrist_Right = 11;
   Hand_Right = 12;
   Hip_Left = 13;
   Knee_Left = 14;
   Ankle_Left = 15;
   Foot_Left = 16; 
   Hip_Right = 17;
   Knee_Right = 18;
   Ankle_Right = 19;
   Foot_Right = 20;
"""
def get_unionData_list(target_list, project_list):

    samples_to_use = []
    for item in target_list:
        if item in project_list:
            samples_to_use.append(item)

    return samples_to_use

class NUCLA_viewProjection(tudata.Dataset):
    """Northeastern-UCLA Dataset Skeleton Dataset
        Access input skeleton sequence, GT label
        When T=0, it returns the whole
    """

    def __init__(self, root_skeleton,root_list, phase, cam, T, target_view='view_1', project_view='view_2', test_view='view_3'):
        self.root_skeleton = root_skeleton
        self.data_root = '/data/Dan/N-UCLA_MA_3D/multiview_action'
        self.root_list = root_list
        self.view = []
        self.phase = phase
        self.target_view = target_view
        self.project_view = project_view
        self.test_view = test_view
        for name_cam in cam.split(','):
            self.view.append('view_' + name_cam)
        self.T = T
        self.num_samples = 0

        self.num_action = 10
        self.action_list = {'a01': 0, 'a02': 1, 'a03': 2, 'a04': 3, 'a05': 4,
                            'a06': 5, 'a08': 6, 'a09': 7, 'a11': 8, 'a12': 9}
        self.actions = {'a01': 'pick up with one hand', 'a02': "pick up with two hands", 'a03': "drop trash",
                        'a04': "walk around", 'a05': "sit down",
                        'a06': "stand up", 'a08': "donning", 'a09': "doffing", 'a11': "throw", 'a12': "carry"}

        # Get the list of files according to cam and phase
        # self.list_samples = []
        self.target_samples = []
        self.project_samples = []
        self.targest_list = []
        self.project_list = []

        self.test_list = []

        # Compute the MEAN and STD of the dataset
        allSkeleton = []
        for view in self.view:
            file_list = os.path.join(self.root_list, f"{view}_{self.phase}.list")
            list_samples = np.loadtxt(file_list, dtype=str)
            for name_sample in list_samples:
                if view == target_view:
                    self.target_samples.append((view, name_sample))
                    self.targest_list.append(name_sample)
                elif view == project_view:
                    self.project_samples.append((view, name_sample))
                    self.project_list.append(name_sample)
                skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)
                if len(skeleton.shape):
                    np.asarray(skeleton)
                allSkeleton.append(skeleton)
        allSkeleton = np.concatenate(allSkeleton, 0)  # ((allSkeleton),0)
        self.x_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 0], 0)  # 1 x num_joint
        self.y_mean_skeleton = np.expand_dims(np.mean(allSkeleton, axis=0)[:, 1], 0)
        self.x_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 0], 0)
        self.y_std_skeleton = np.expand_dims(np.std(allSkeleton, axis=0)[:, 1], 0)

        self.samples_list = []
        for item in self.targest_list:
            if item in self.project_list:
                self.samples_list.append(item)

        if self.phase == 'test':
            # self.test_list = []
            test_file1 = np.loadtxt(os.path.join(self.root_list, f"{self.test_view}_train.list"), dtype=str)
            test_file2 = np.loadtxt(os.path.join(self.root_list, f"{self.test_view}_val.list"), dtype=str)
            test_file3 = np.loadtxt(os.path.join(self.root_list, f"{self.test_view}_test.list"), dtype=str)

            self.test_list = np.concatenate((test_file1, test_file2, test_file3))

            self.samples_list = self.test_list

        elif self.phase == 'val':
            self.val_list = []
            val_file1 = np.loadtxt(os.path.join(self.root_list, f"{self.target_view}_val.list"), dtype=str)

            for item in val_file1:
                self.val_list.append((self.target_view, item))
            val_file2 = np.loadtxt(os.path.join(self.root_list, f"{self.project_view}_val.list"), dtype=str)
            for item in val_file2:
                self.val_list.append((self.project_view, item))

            # self.val_list = np.concatenate((val_file1, val_file2))
            self.samples_list = self.val_list

        else:
            self.samples_list = self.samples_list

        # Print
        # print(f"=== {self.phase} {self.view}: {len(self.project_samples) + len(self.target_samples)} samples ===")
        # print(f"==={target_view}:{len(self.target_samples)} samples ===")
        # print(f"==={project_view}:{len(self.project_samples)} samples ===")
        # print(f"=== samples to use:{len(self.samples_list)} samples ===")
        print(f"==={self.phase}: {len(self.samples_list)}samples ===")


    def __len__(self):
      return len(self.samples_list)

    def get_norm(self, skeleton):
        '''Get the normalization according to the skeleton
        Arguments:
            skeleton: T(8) x num_joints(25) x 2, unnormalized in original resolution
        Return:
            skeleton: T(8) x num_joints(25) x 2, normalized in original resolution
        '''
        X = skeleton[:, :, 0]
        Y = skeleton[:, :, 1]
        normX = (X - self.x_mean_skeleton) / self.x_std_skeleton
        normY = (Y - self.y_mean_skeleton) / self.y_std_skeleton
        normSkeleton = np.concatenate((np.expand_dims(normX, 2), np.expand_dims(normY, 2)), 2).astype(
            float)  # inputLen x 15 x 2

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
        meanX_mat = torch.FloatTensor(self.x_mean_skeleton).repeat(framNum, 1)  # inputLen x 15
        meanY_mat = torch.FloatTensor(self.y_mean_skeleton).repeat(framNum, 1)
        stdX_mat = torch.FloatTensor(self.x_std_skeleton).repeat(framNum, 1)
        stdY_mat = torch.FloatTensor(self.y_std_skeleton).repeat(framNum, 1)

        X = normSkeleton[:, :, 0]  # inputLen x 15
        Y = normSkeleton[:, :, 1]  # inputLen x 15
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
                transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

            img_tensor = preprocess(input_image)

            imgSequence.append(img_tensor.unsqueeze(0))

        imgSequence = torch.cat((imgSequence), 0)

        return imgSequence, imgSize

    def get_data(self, view, name_sample):

        skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)
        T_sample, num_joints, dim = skeleton.shape

        if self.T == 0:
            skeleton_input = skeleton
            # imgSequence = np.zeros((T_sample, 3, 224, 224))
            details = {'name_sample': name_sample, 'T_sample': T_sample, 'time_offset': range(T_sample), 'view':view}
        else:
            stride = T_sample / self.T
            ids_sample = []
            for i in range(self.T):
                id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
                ids_sample.append(id_sample)

            skeleton_input = skeleton[ids_sample, :, :]
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
        imageSequence, _ = self.get_grayscale(view, name_sample)

        skeletonData = {'normSkeleton':normSkeleton, 'unNormSkeleton':unNormSkeleton}
        return heatmap_to_use, imageSequence,skeletonData, details

    def __getitem__(self, index):
        """
        Return:
            skeletons: FloatTensor, [T, num_joints, 2]
            label_action: int, label for the action
            info: dict['sample_name', 'T_sample', 'time_offset']
        """

        if self.phase == 'val':


            view, name_sample = self.samples_list[index]

            label_action = self.action_list[name_sample[:3]]  # 0-9

            val_view_heat, val_view_image, val_view_skeleton, val_info = self.get_data(view, name_sample)
            dicts = {'heatmaps':val_view_heat, 'imageSequence':val_view_image, 'skeletonSeq':val_view_skeleton, 'info':val_info, 'action':label_action}

        else:
            name_sample = self.samples_list[index]
            label_action = self.action_list[name_sample[:3]]

            if self.phase == 'test':
                test_view_heat, test_view_image, test_view_skeleton, test_info = self.get_data(self.test_view,
                                                                                                       name_sample)

                dicts = {'test_view_heat':test_view_heat, 'test_view_image':test_view_image, 'test_view_skeleton':test_view_skeleton,
                         'test_info':test_info, 'action':label_action}
            else:
                target_view_heat, target_view_image, target_view_skeleton, target_info = self.get_data(self.target_view, name_sample)

                project_view_heat, project_view_image, projet_view_skeleton, project_info = self.get_data(self.project_view, name_sample)

                dicts = {'target_view_heat': target_view_heat, 'target_view_image':target_view_image, 'target_view_skeleton':target_view_skeleton, 'target_info': target_info,
                         'project_view_heat': project_view_heat, 'project_view_image':project_view_image, 'project_view_skeleton':projet_view_skeleton, 'project_info':project_info
                         ,'action':label_action}


        return dicts


if __name__ == "__main__":

    root_skeleton = '/data/Dan/N-UCLA_MA_3D/skeletons_res'
    root_list = f"/data/Dan/N-UCLA_MA_3D/lists"
    DS = NUCLA_viewProjection(root_skeleton=root_skeleton, root_list=root_list, phase='test', cam='1,2', T=0, target_view='view_1',
                              project_view='view_2', test_view='view_3')

    dataloader = tudata.DataLoader(DS, batch_size=1, shuffle=False,
                                   num_workers=1, pin_memory=True)
    print(len(DS))
    for i, sample in enumerate(dataloader):
        print(i)
        pass
