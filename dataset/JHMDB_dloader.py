import numpy as np
import torch
from PIL import Image,ImageFilter,ImageEnhance
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from h5py import File
from utils import Gaussian, DrawGaussian
import os
import scipy.io
import statistics
import random
from six.moves import xrange
import fnmatch
import json
import glob

# borrowed from: https://github.com/lawy623/LSTM_Pose_Machines/blob/master/dataset/JHMDB/JHMDB_PreData.m
# order
# 0: neck    1:belly   2: face
# 3: right shoulder  4: left shoulder
# 5: right hip       6: left hip
# 7: right elbow     8: left elbow
# 9: right knee      10: left knee
# 11: right wrist    12: left wrist
# 13: right ankle    14: left ankle

def get_train_test_annotation(dataRoot):

    subFolder = os.path.join(dataRoot, 'sub_splits')
    imageFolder = os.path.join(dataRoot, 'Rename_Images')
    maskFolder = os.path.join(dataRoot, 'puppet_mask')
    poseFolder = os.path.join(dataRoot, 'joint_positions')

    baselineFolder = os.path.join(dataRoot, 'baseline_resnet101')
    # baselineFolder = os.path.join(dataRoot, 'openpose_joints')
    totTXTlist = os.listdir(subFolder)
    actionList = {'golf':0, 'kick_ball':1, 'run':2, 'climb_stairs':3, 'catch':4, 'pick':5, 'jump':6, 'shoot_ball':7,
                  'walk':8, 'pullup':9, 'push':10, 'swing_baseball':11}


    trainAnnot = []
    testAnnot = []
    for i in range(0, len(totTXTlist)):
        filename = os.path.join(subFolder, totTXTlist[i])
        action = totTXTlist[i].split('_test_')[0]
        label = actionList[action]
        with open(filename) as f:
            content = f.readlines()

        for t in range(0, len(content)):

            folder_to_use = content[t].split('\n')[0].split('.avi')[0]
            traintest = content[t].split('\n')[0].split('.avi')[1]   # 1: train; 2: test

            imgPath = os.path.join(imageFolder, action, folder_to_use)
            posePath = os.path.join(poseFolder, action, folder_to_use)
            maskPath = os.path.join(maskFolder, action, folder_to_use)


            annot = scipy.io.loadmat(os.path.join(posePath, 'joint_positions'))
            bbox = scipy.io.loadmat(os.path.join(maskPath, 'Bbox.mat'))['Bbox']
            mask = scipy.io.loadmat(os.path.join(maskPath, 'puppet_mask.mat'))['part_mask']
            'conver openpose-25 to openpose-18'
            # baselinePath = os.path.join(baselineFolder, action, folder_to_use)
            # baseline = get_openpose(baselinePath)



            # dicts = {'imgPath': imgPath, 'annot': annot, 'Bbox': bbox, 'mask': mask, 'baseline': baseline,
            #          'actLabel': label}
            if int(traintest) == 1:
                dicts = {'imgPath': imgPath, 'annot': annot, 'Bbox': bbox, 'mask': mask, 'baseline': None, 'actLabel':label}
                trainAnnot.append(dicts)
            else:
                baselinePath = os.path.join(baselineFolder, folder_to_use)
                baseline = scipy.io.loadmat(os.path.join(baselinePath, 'preds.mat'))['preds']

                dicts = {'imgPath': imgPath, 'annot': annot, 'Bbox': bbox, 'mask': mask, 'baseline': baseline,
                                  'actLabel': label}
                testAnnot.append(dicts)

    return trainAnnot, testAnnot, actionList

def get_openpose(pose_dir):
    poses_names = glob.glob(os.path.join(pose_dir, '*.json'))
    poses_names.sort()
    temp = []
    for i in range(0, len(poses_names)):
        f = open(poses_names[i])
        data = json.load(f)
        points = data['people']

        # print('check')
        if points != []:
            # for p in range(0, len(points)):
            if len(points) == 1:
                point = np.asarray(points[0]['pose_keypoints_2d'])  # list
                index = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
                pose = np.reshape(point, (25, 3))[index,:] # 2d joints
                temp.append(pose)

    pose = np.asarray(temp)

    return pose


class jhmdbDataset(data.Dataset):
    def __init__(self, trainAnnot, testAnnot, T, split, if_occ):
        self.trainSet = trainAnnot[0:600]
        self.testSet = testAnnot
        self.valSet = trainAnnot[600:]
        self.inputLen = T
        self.split = split
        self.numJoint = 15
        self.clips = 6
        self.if_occ = if_occ
        if self.split == 'train':
            self.dataLen = len(self.trainSet)
        elif self.split == 'val':
            self.dataLen = len(self.valSet)
        else:
            self.dataLen = len(self.testSet)

        numData = len(self.trainSet)
        allSkeleton = []
        for i in range(0, numData):
            skeleton = self.trainSet[i]['annot']['pos_img']
            allSkeleton.append(skeleton)

        allSkeleton = np.concatenate((allSkeleton), 2)
        self.meanX = np.expand_dims(np.mean(allSkeleton, axis=2)[0], 0)  # 1 x num_joint
        self.meanY = np.expand_dims(np.mean(allSkeleton, axis=2)[1], 0)
        self.stdX = np.expand_dims(np.std(allSkeleton, axis=2)[0],0)
        self.stdY = np.expand_dims(np.std(allSkeleton, axis=2)[1],0)

    def __len__(self):
        return self.dataLen

    def read_annot(self, annotSet):
        imgPath = annotSet['imgPath']
        Bbox = annotSet['Bbox']
        skeleton = annotSet['annot']['pos_img'].transpose(2, 1, 0)     # 2 x 15 x T ---> T x 15 x 2
        img_mask = annotSet['mask']
        # mn = np.min(skeleton, 1)
        # mx = np.max(skeleton, 1)
        # Bbox = np.concatenate((mn, mx-mn+1), 1)
        # Bbox = np.concatenate((mn, mx), 1)
        if annotSet['baseline'] is None:
            baseline = np.zeros(skeleton.shape)
        else:
            baseline = annotSet['baseline']

        actLabel = annotSet['actLabel']

        return imgPath, Bbox, skeleton, baseline, actLabel

    def preProcessImage(self, imgPath, Bbox):
        imgList = fnmatch.filter(os.listdir(imgPath), '*.png')
        imgList.sort()

        # imgList.remove('.')
        if self.if_occ:
            frame_occ_idx = self.get_occlusion_idx(len(imgList), occRatio=0.1)
        else:
            frame_occ_idx = self.get_occlusion_idx(len(imgList), occRatio=0.0)
        imgSequence = []
        imgSize = []
        for i in range(0, len(imgList)):
            img_path = os.path.join(imgPath, imgList[i])
            h = Bbox[i, 2] - Bbox[i, 0]
            w = Bbox[i, 3] - Bbox[i, 1]
            c1 = np.expand_dims(Bbox[i, 0] + 0.5 * h, 0)
            c2 = np.expand_dims(Bbox[i, 1] + 0.5 * w, 0)
            center = np.concatenate((c1, c2), 0)

            input_image = Image.open(img_path)
            imgSize.append(input_image.size)

            if i in frame_occ_idx:

                input_image1 = self.manuallyOcclusion(input_image, center)
                input_image2 = self.get_blurImage(input_image)
                input_image3 = self.change_Brightness(input_image)
                print('check')

            else:
                input_image = input_image

            # bbx = Bbox[i]
            # cropedwithBbox = input_image.crop(bbx)

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

    def get_random_crop(self, input_image, crop_height, crop_width):

        input_image = np.asarray(input_image)
        max_x = input_image.shape[1] - crop_width
        max_y = input_image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        cropImg = input_image[y: y + crop_height, x: x + crop_width]
        cropImg  = Image.fromarray(cropImg)

        return cropImg

    def get_occlusion_idx(self, T, occRatio):
        random.seed(1234567890)
        n = int(T * occRatio)
        # print('occluded frame number:', n, 'frame num:', T)
        frameIdx = list(np.linspace(0, T - 1, T-1, dtype=np.int))
        if occRatio < 1.0:
            frame_occ_idx = random.sample(frameIdx, k=n)
        else:
            frame_occ_idx = frameIdx

        return frame_occ_idx

    def manuallyOcclusion(self, input_image, center):
        'manually occlusion'
        random.seed(1234567890)
        # input_image_occ = input_image
        # Normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_image = np.array(input_image)
        psize = 80
        occlusion = np.zeros((psize, psize, 3))  # RGB

        x = np.random.randint(center[0] - 5, center[0] + 5)
        y = np.random.randint(center[1] - 5, center[1] + 5)

        # input_image[y - int(psize / 2):y + int(psize / 2), x - int(psize / 2):x + int(psize / 2)] = occlusion
        input_image[y - int(psize / 2):y + int(psize / 2), x - int(psize / 2):x + int(psize / 2)] = 0
        input_image = Image.fromarray(input_image)
        return input_image

    def get_blurImage(self, input_image):
        blurImage =  input_image.filter(ImageFilter.GaussianBlur(radius=4))

        return blurImage

    def change_Brightness(self, input_image):
        random.seed(1234567890)
        x = random.choice([2, 3, 4, 5])
        y = random.choice([0.1, 0.2, 0.3, 0.4])
        n = random.choice([x, y])  # either brighter or darker

        enhancer = ImageEnhance.Brightness(input_image)
        enhancedImage = enhancer.enhance(n)

        return enhancedImage


    def data_to_use(self, skeleton, bbox, imgSequence):
        nframes = imgSequence.shape[0]
        random.seed(1234567890)
        useLen = self.inputLen

        if len(skeleton.shape) == 4:
            skeleton = skeleton.squeeze(0)  # inputLen x 15 x 2
            bbox = bbox.squeeze(0)  # T x 4
            imgSequence = imgSequence.squeeze(0)
            # baselineSkeleton = baselineSkeleton.squeeze(0)

        if nframes > useLen:
            idx = random.randint(0, nframes - useLen)
            # data_sel = skeleton[idx: idx + useLen].unsqueeze(0)  # 1 x useLen x 15 x 2
            # bbx_sel = bbox[idx: idx + useLen].unsqueeze(0)  # 1 x inputLen x 4
            # img_sel = imgSequence[idx: idx + useLen]

            data_sel = np.expand_dims(skeleton[idx: idx + useLen], 0)
            bbx_sel = np.expand_dims(bbox[idx: idx + useLen], 0)
            img_sel = np.expand_dims(imgSequence[idx: idx + useLen], 0)
            # base_sel = np.expand_dims(baselineSkeleton[idx: idx + useLen], 0)

            sequence_to_use = data_sel
            bbox_to_use = bbx_sel
            imgSequence_to_use = img_sel
            mask_idx = np.ones((sequence_to_use.shape))
            # baseline_to_use = base_sel


        elif nframes == useLen:
            idx = 0
            sequence_to_use = skeleton
            # baseline_to_use = baselineSkeleton
            mask_idx = np.ones((sequence_to_use.shape))
            if bbox.shape[0] == nframes:
                bbox_to_use = bbox
                imgSequence_to_use = imgSequence

            else:
                bbox_to_use = bbox[0:nframes]   # need to check'
                imgSequence_to_use = imgSequence[0:nframes]


        else:
            seqLeft = useLen - nframes
            sequence = []
            # base = []
            img_sequence = []
            bbx = []
            m_idx = []
            idx = 0   # start from first frame
            for i in xrange(seqLeft):
                # data_sel = skeleton[-1]   # last frame
                #
                # bbx_sel = bbox[-1,:]
                mask_sel = np.zeros((self.numJoint, 2))
                data_sel = np.zeros((self.numJoint, 2))
                # base_sel = np.zeros((self.numJoint, 2))   # baseline joint is 14
                bbx_sel = np.zeros((4))
                img_sel = torch.zeros(3, 224, 224)

                sequence.append(np.expand_dims(data_sel, 0))
                # base.append(np.expand_dims(base_sel, 0))
                bbx.append(np.expand_dims(bbx_sel, 0))
                img_sequence.append(img_sel.unsqueeze(0))
                m_idx.append(np.expand_dims(mask_sel, 0))

            sequence = np.concatenate(sequence, axis=0)   # seqLeft x 15 x 2
            bbx = np.concatenate(bbx, axis=0)  # seqLeft x 4
            sequence_img = torch.cat((img_sequence), 0)
            ma_idx = np.concatenate(m_idx, axis=0)
            # base = np.concatenate(base, axis=0)


            # sequence_to_use = torch.Tensor(np.concatenate((skeleton.numpy(), sequence), axis=0)).type(torch.FloatTensor).unsqueeze(0)  # 1 x inputLen x 15 x 2
            # bbox_to_use = torch.Tensor(np.concatenate((bbox.numpy(), bbx), axis=0)).type(torch.FloatTensor).unsqueeze(0)

            sequence_to_use = np.concatenate((skeleton, sequence), axis=0).astype(float)
            # baseline_to_use = np.concatenate((baselineSkeleton, base), axis=0).astype(float)
            mask_part1 = np.ones((skeleton.shape))
            mask_idx = np.concatenate((mask_part1, ma_idx), axis=0).astype(float)
            if bbox.shape[0] == nframes:
                bbox_to_use = np.concatenate((bbox, bbx), axis=0).astype(float)
                imgSequence_to_use = torch.cat((imgSequence, sequence_img), 0).type(torch.FloatTensor)
            else:
                bbox_to_use = np.concatenate((bbox[0:nframes], bbx), axis=0).astype(float)
                imgSequence_to_use = torch.cat((imgSequence[0:nframes], sequence_img), 0).type(torch.FloatTensor)

        return sequence_to_use, bbox_to_use, imgSequence_to_use, mask_idx, nframes, idx

    def get_normalized_data(self, skeleton):
        'skeleton : 15 x 2 x T'

        # skeleton = torch.Tensor(annotSet['annot']['pos_img']).permute(2, 1, 0).type(torch.FloatTensor)
        # nframes = skeleton.shape[2]
        # skeleton = sequence_to_use.squeeze(0)

        X = skeleton[:,:,0]
        Y = skeleton[:,:,1]

        # meanX_mat = torch.FloatTensor(self.meanX).repeat(1, nframes)
        # meanY_mat = torch.FloatTensor(self.meanY).repeat(1, nframes)
        # stdX_mat = torch.FloatTensor(self.stdX).repeat(1, nframes)
        # stdY_mat = torch.FloatTensor(self.stdY).repeat(1, nframes)
        # normX = (X - meanX_mat)/stdX_mat
        # normY = (Y - meanY_mat)/stdY_mat
        #
        # normSkeleton = torch.cat((normX.unsqueeze(0), normY.unsqueeze(0)), 0).type(torch.FloatTensor)

        normX = (X - self.meanX)/self.stdX
        normY = (Y - self.meanY)/self.stdY

        normSkeleton = np.concatenate((np.expand_dims(normX,2), np.expand_dims(normY,2)), 2).astype(float) # inputLen x 15 x 2

        return normSkeleton

    def get_normalized_data_openpose(self, skeleton, imageSize):
        normSkeleton = np.zeros((skeleton.shape))
        for i in range(0, skeleton.shape[0]):
            normSkeleton[i,:,0] = skeleton[i,:,0]/imageSize[i][0]
            normSkeleton[i,:,1] = skeleton[i,:,1]/imageSize[i][1]
            normSkeleton[i,:,2] = skeleton[i,:,2]

        return normSkeleton

    def get_unNormalized_data(self, normSkeleton):
        'for inference part, normSkeleton : N x useLen x 15 x 2'

        # nframes = normSkeleton.shape[2]
        if len(normSkeleton.shape) == 4:
            normSkeleton = normSkeleton.squeeze(0)

        if isinstance(normSkeleton, np.ndarray):
            normSkeleton = torch.Tensor(normSkeleton).float()

        framNum = normSkeleton.shape[0]
        meanX_mat = torch.FloatTensor(self.meanX).repeat(framNum, 1)   # inputLen x 15
        meanY_mat = torch.FloatTensor(self.meanY).repeat(framNum, 1)
        stdX_mat = torch.FloatTensor(self.stdX).repeat(framNum, 1)
        stdY_mat = torch.FloatTensor(self.stdY).repeat(framNum, 1)

        X = normSkeleton[:,:,0]  # inputLen x 15
        Y = normSkeleton[:,:,1]  # inputLen x 15

        unNormX = X * stdX_mat + meanX_mat
        unNormY = Y * stdY_mat + meanY_mat

        unNormSkeleton = torch.cat((unNormX.unsqueeze(2), unNormY.unsqueeze(2)), 2)

        return unNormSkeleton

    def pose_to_heatmap(self, poses, image_size, outRes):
        'input(joints): T x njoints x 2'
        'output(heatmaps): T x 64 x 64 '

        GaussSigma = 1

        # T = poses.shape[0]
        H = image_size[0]
        W = image_size[1]
        heatmaps = []
        for t in range(0, self.inputLen):
            pts = poses[t]  # njoints x 2
            out = np.zeros((pts.shape[0], outRes, outRes))
            # out = []
            for i in range(0, pts.shape[0]):
                pt = pts[i]
                if pt[0] == 0 and pt[1] == 0:
                    out[i] = np.zeros((outRes, outRes))
                # path = np.zeros((imgSize))
                else:
                    newPt = np.array([outRes * (pt[0] / W), outRes * (pt[1] / H)])

                    out[i] = DrawGaussian(out[i], newPt, GaussSigma)
                # out.append(hm)
            # out = np.stack(out, axis=0)
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


    def get_data_multi(self, normSkeleton, imageSequence):
        # imageSequence, _, imageSequence_orig = self.get_rgb(view, name_sample)
        # if self.dataType == '2D':
        #     skeleton, usedID = getJsonData(os.path.join(self.root_skeleton, view), name_sample)
        #     imageSequence = imageSequence[usedID]
        #     normSkeleton = self.get_norm(skeleton)
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
        # else:
        #     skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)
        #     normSkeleton = self.get_norm(skeleton)
        #     T_sample, num_joints, dim = normSkeleton.shape
        #
        #     if T_sample < self.clips:
        #         Tadd = abs(T_sample - self.clips)
        #         last = np.expand_dims(normSkeleton[-1, :, :], 0)
        #         copyLast = np.repeat(last, Tadd, 0)
        #         normSkeleton = np.concatenate((normSkeleton, copyLast), 0)  # copy last frame Tadd times
        #         T_sample = self.clips
        #
        #     stride = T_sample / self.clips
        #     ids_sample = []
        #     for i in range(self.clips):
        #         id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
        #         ids_sample.append(id_sample)
        #     L = 36  # clip length
        #     if T_sample <= L:
        #         temp = np.expand_dims(normSkeleton, 0)
        #         allSkeletons = np.repeat(temp, self.clips, 0)
        #         allImageSequence = np.zeros((self.clips, T_sample, 3, 224, 224))
        #
        #     else:
        #         allSkeletons = []
        #         for id in ids_sample:
        #
        #             if id - int(L / 2) < 0 and T_sample - (id + int(L / 2)) >= 0:
        #                 temp = np.expand_dims(normSkeleton[0:L], 0)
        #
        #             elif id - int(L / 2) >= 0 and T_sample - (id + int(L / 2)) >= 0:
        #                 temp = np.expand_dims(normSkeleton[id - int(L / 2):id + int(L / 2)], 0)
        #
        #
        #             elif id - int(L / 2) >= 0 and T_sample - (id + int(L / 2)) < 0:
        #                 temp = np.expand_dims(normSkeleton[T_sample - L:], 0)
        #
        #
        #             allSkeletons.append(temp)
        #
        #         allSkeletons = np.concatenate((allSkeletons), 0)
        #         allImageSequence = np.zeros((self.clips, L, 3, 224, 224))



        return allSkeletons, allImageSequence


    def __getitem__(self, idx):
        if self.split == 'train':
            annotSet = self.trainSet[idx]

        elif self.split == 'val':
            annotSet = self.valSet[idx]
        else:
            annotSet = self.testSet[idx]

        imgPath, Bbox, gtSkeleton, baseline, actLabel = self.read_annot(annotSet)

        if self.split == 'test':

            # normSkeleton = self.get_normalized_data(baseline)
            normSkeleton = self.get_normalized_data(gtSkeleton)
        else:
            normSkeleton = self.get_normalized_data(gtSkeleton)

        imgSequence, imageSize = self.preProcessImage(imgPath, Bbox)

        allSkeletons, allImageSequence = self.get_data_multi(normSkeleton, imgSequence)
        dicts = {'testImgSequence': allImageSequence,
                 'testSkeleton': allSkeletons, 'actLabel':actLabel}



        # imgSequence = self.manuallyOcclusion(imgSequence)

        # baseline_to_use = self.get_normalized_data(baseline)
        # baseline_to_use = self.get_normalized_data_openpose(baseline, imageSize)


        # baselineSkeleton = self.get_normalized_data(self.decompose_baseline(baseline, Skeleton))
        # baselineSkeleton = self.decompose_baseline(baseline, Skeleton)

        # sequence_to_use, Bbox_to_use, imgSequence_to_use, mask_idx, nframes, idx = self.data_to_use(normSkeleton, Bbox, imgSequence)
        # # print('skeleton shape:', sequence_to_use.shape, 'baseline shape:', baseline.shape)
        #
        # imageSize = [imgSequence.shape[2], imgSequence.shape[3]]
        #
        # unNormSkeleton = self.get_unNormalized_data(sequence_to_use)
        # heatmap_to_use = self.pose_to_heatmap(unNormSkeleton, imageSize, 64)
        #
        # """""
        # filename = './debugFile/heatmap.jpg'
        # plt.imsave(filename, heatmap_to_use[0], cmap='seismic', format='jpg')
        #
        # rgb_name = './debugFile/rgb.jpg'
        # image = imgSequence_to_use[0,(2,1,0),:,:].numpy().transpose(1, 2, 0)
        # images = (255*(images+1))/2
        # cv2.imwrite(rgb_name, images)
        #
        #
        # """""
        #
        # dicts = {'imgSequence_to_use': imgSequence_to_use, 'Bbox_to_use': Bbox_to_use,
        #          'sequence_to_use': sequence_to_use, 'mask_idx': mask_idx, 'nframes':nframes,
        #          'randomInd:':idx, 'imgPath':imgPath, 'heatmap_to_use':heatmap_to_use, 'actLabel':actLabel}
        #



        return dicts


if __name__ =='__main__':
    dataRoot = '/data/Yuexi/JHMDB'

    trainAnnot, testAnnot, actionList = get_train_test_annotation(dataRoot)

    N = np.zeros(len(testAnnot))
    for i in range(0, len(testAnnot)):
        nframes = testAnnot[i]['annot']['pos_img'].shape[2]

        N[i] = (nframes)

    dataset = jhmdbDataset(trainAnnot, testAnnot, T=40, split='train', if_occ=False)
    dloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    for i, sample in enumerate(dloader):
        label = sample['actLabel']
        testImgSequence = sample['testImgSequence']
        testSkeleton = sample['testSkeleton']
        print('label:', label, testSkeleton.shape, testImgSequence.shape)

        # sequence_to_use = sample['sequence_to_use']
        # Bbox = sample['Bbox_to_use']
        # imgSequence_to_use = sample['imgSequence_to_use']
        # mask_idx = sample['mask_idx']
        # print('sample:', i, 'squence shape:', sequence_to_use.shape,
        #       'bbox shape:' , Bbox.shape, 'imgSeq:', imgSequence_to_use.shape, 'mask_idx:', mask_idx)
    print('check')