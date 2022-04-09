import os
import math
import numpy as np
import pickle
import random
import torch
import torch.utils.data as dataset
from scipy.sparse import rand
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import cv2


def generateData(num_sample, Npole):
    random.seed(0)
    # C1 = 100*torch.rand((num_sample, Npole, 1))
    # C2 = 100*torch.rand((num_sample, Npole, 1))
    C1 = torch.zeros((num_sample, Npole, 1))
    C2 = torch.zeros((num_sample, Npole, 1))

    return C1, C2

def generateSparseMatrix(num_sample,Npole, dim):
    random.seed(123)
    X = np.zeros((num_sample, Npole, dim))
    for i in range(0, num_sample):
        x = rand(Npole, dim, density=0.1, format='csr')
        x.data[:] = 1
        x = x.todense()
        for d in range(0, dim):
            temp = x[:,d]
            nonZeros = np.transpose(np.nonzero(temp)).shape[0]
            l = np.linspace(0, nonZeros-1, nonZeros, dtype=np.int)
            ZEROS = np.linspace(1, nonZeros-2, nonZeros-2+1, dtype=np.int)
            zeroNum = random.sample(list(ZEROS), 1)[0]
            negativeID = random.sample(list(l), zeroNum)
            temp[negativeID] = -1
            x[:, d] = temp

        X[i] = x
    return X

def getMultiClassData(num_sample, Npole, dim):
    Class0 = generateSparseMatrix(num_sample, Npole, dim)
    Class1 = generateSparseMatrix(num_sample, Npole, dim)
    Class2 = generateSparseMatrix(num_sample, Npole, dim)
    Class3 = generateSparseMatrix(num_sample, Npole, dim)
    Class4 = generateSparseMatrix(num_sample, Npole, dim)

    data = {'class0':Class0, 'class1':Class1, 'class2':Class2,
            'class3':Class3, 'class4':Class4}
    return data


class generateSytheticData_MultiClass(dataset.Dataset):
    def __init__(self, data, num_Sample, Npole, dim, phase):
        self.num_sample = num_Sample
        self.Npole = Npole
        self.dim = dim
        self.phase = phase
        # self.action = ['class0', 'class1', 'class2', 'class3', 'class4']
        self.action = ['class0', 'class1', 'class2', 'class3']
        self.testAction = ['class0']
        # self.cls = [0, 1]
        self.data = data

        if self.phase == 'train':
            self.Index = np.linspace(0, int(0.8*self.num_sample-1), int(0.8*self.num_sample),dtype=np.int)
        elif self.phase == 'val':
            self.Index = np.linspace(int(8*self.num_sample), int(0.9*self.num_sample-1),int(0.1*self.num_sample),dtype=np.int)
        else:
            # self.Index = np.linspace(int(0.9*self.num_sample), self.num_sample-1,int(0.1*self.num_sample),dtype=np.int)
            self.Index = np.linspace(0, self.num_sample-1, self.num_sample)

    def __len__(self):
        return len(self.Index)

    def __getitem__(self, idx):
        random.seed(123)
        noise = torch.randn((self.Npole, self.dim))
        value = 100 * torch.rand((self.Npole, self.dim))
        if self.phase == 'test':
            action = self.testAction
            c = self.data[action][idx]
        else:
            action = random.choice(self.action)

            c = self.data[action][idx]

        inputData = np.multiply(value, c) + noise
        sample = {'input': inputData.unsqueeze(-1), 'class': action}

        return sample

    """
    def __getitem__(self, idx):
        cls = random.choice(self.cls)
        noise1 = torch.randn((self.Npole, self.dim))
        value1 = 100 * torch.rand((self.Npole, self.dim))

        noise2 = torch.randn((self.Npole, self.dim))
        value2 = 100 * torch.rand((self.Npole, self.dim))

        if cls == 0:
            act = random.choice(self.action)
            c = self.data[act][idx]
            view1 = np.multiply(value1, c) + noise1
            view2 = np.multiply(value2, c) + noise2

            sample = {'class':cls, 'view1':view1, 'view2':view2, 'act1':act, 'act2':act}
        else:
            # action = self.action
            act = random.sample(self.action,2)
            act1 = act[0]
            act2 = act[1]

            c1 = self.data[act1][idx]
            c2 = self.data[act2][idx]
            view1 = np.multiply(value1, c1) + noise1
            view2 = np.multiply(value2, c2) + noise2
            sample = {'class': cls, 'view1': view1, 'view2': view2, 'act1': act1, 'act2':act2}

        return sample
    """

class generateSytheticData(dataset.Dataset):
    def __init__(self, view1Data, view2Data, Npole, phase):
        # self.BS = BatchSize
        self.Npole = Npole
        self.num_sample = view1Data.shape[0]
        self.view1Data = view1Data
        self.view2Data = view2Data
        self.phase = phase
        self.nonZeroID1 = [10, 45, 87, 96, 102, 145]
        self.nonZeroID2 = [17, 38, 54, 99, 132, 160]

        if self.phase == 'train':
            self.C1 = self.view1Data[0:int(0.8*self.num_sample), :, :]
            self.C2 = self.view2Data[0:int(0.8*self.num_sample):, :, :]
        elif self.phase == 'val':
            self.C1 = self.view1Data[int(0.8*self.num_sample):int(0.9*self.num_sample), :, :]
            self.C2 = self.view2Data[int(0.8*self.num_sample):int(0.9*self.num_sample), :, :]
        else:
            self.C1 = self.view1Data[int(0.9*self.num_sample):, :, :]
            self.C2 = self.view2Data[int(0.9*self.num_sample):, :, :]

    def __len__(self):
        # if self.phase == 'train':
        return self.C1.shape[0]
    def __getitem__(self, idx):
        clsID = [0, 1]  # 0: same class, 1: different class
        cls = random.choice(clsID)
        l = np.linspace(0, self.Npole - 1, num=self.Npole, dtype=np.int)
        s = 120
        c1 = self.C1[idx]
        c2 = self.C2[idx]
        if cls == 0:
            negativeID = random.sample(self.nonZeroID1, 2)
            c1[self.nonZeroID1, :] = 100 * (torch.rand((6, 1)))
            view1 = c1 + torch.randn(c1.shape)
            view1[negativeID, :] = -1 * view1[negativeID]
            sample = {'input': view1.unsqueeze(-1), 'class': cls}
        else:
            negativeID = random.sample(self.nonZeroID2, 3)
            c2[self.nonZeroID2, :] = 100 * (torch.rand((6, 1)))
            view2 = c2 + torch.randn(c2.shape)
            view2[negativeID, :] = -1 * view2[negativeID, :]
            sample = {'input': view2.unsqueeze(-1), 'class': cls}

        return sample


class generatedSyntheticData_Gumbel(dataset.Dataset):
    def __init__(self, dictionary, nonZeroID,nonZeroValue, Npole, num_sample, phase):
        self.Npole = Npole
        self.num_sample = num_sample
        self.phase = phase
        self.dictionary = dictionary
        self.nonZeroID = nonZeroID
        self.nonZeroValue = nonZeroValue
        if self.phase == 'train':
            self.coeff = torch.zeros((int(0.8*self.num_sample),self.Npole, 1))
        else:
            self.coeff = torch.zeros((int(0.2*self.num_sample), self.Npole, 1))

    def __len__(self):
        return self.coeff.shape[0]

    def __getitem__(self, idx):
        # negativeID = random.sample(self.nonZeroID, 2)
        c = self.coeff[idx]
        c[self.nonZeroID,:] = self.nonZeroValue[idx]
        y = torch.matmul(self.dictionary, c)
        bi = torch.zeros(c.shape)
        bi[self.nonZeroID] = 1
        dict = {'input':y, 'coeff':c, 'binary':bi}
        return dict


if __name__ == '__main__':
    Npole = 161
    num_Sample = 100

    """""
    # trainSet = generateSytheticData(view1Data=view1Data, view2Data=view2Data, Npole=161, phase='train')
    data = getMultiClassData(num_Sample, 161, 36)
    trainSet = generateSytheticData_MultiClass(data=data, num_Sample=100, Npole=161, dim=36, phase='train')
    trainloader = dataset.DataLoader(trainSet, batch_size=1, shuffle=True,
                                   num_workers=4, pin_memory=True)

    count_cls0 = 0
    count_cls1 = 0
    for i, sample in enumerate(trainloader):
        inputData = sample['input']
        print(inputData.shape)
        
        if sample['class'].data.item() == 0:
            count_cls0 +=1
        elif sample['class'].data.item() == 1:
            count_cls1 +=1

        # print('check')

    print('# cls0:', count_cls0, '# cls1:', count_cls1)
    print('done')
    """
    dictionary = torch.randn(36, 161)
    nonZeroID = [10, 33, 64, 73, 90, 151]
    trainSet = generatedSyntheticData_Gumbel(dictionary=dictionary, nonZeroID=nonZeroID, Npole=Npole, num_sample=num_Sample, phase='train')
    trainloader = dataset.DataLoader(trainSet, batch_size=10, shuffle=True, num_workers=4, pin_memory=True)

    for i, sample in enumerate(trainloader):
        input = sample['input']
        c = sample['coeff']
        print('check')