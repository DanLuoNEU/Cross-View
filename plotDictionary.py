import numpy as np
from torch.utils.data import DataLoader
import scipy.io
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from dataset.NUCLA_ViewProjection_trans import *
import torchvision
from modelZoo.BinaryCoding import *
import os
def plotpoles(r, theta):
    # r = model.l1.rr.data.cpu().numpy()                       #r should be numpy array with size (20,)
    # theta = model.l1.theta.data.cpu().numpy()                     #same as r
    r = r.cpu().numpy()
    theta = theta.cpu().numpy()
    ax = plt.subplot(1,1,1, projection='polar')
    ax.scatter(0,1,c='black', s = 5)
    ax.scatter(theta, r,s = 5)
    ax.scatter(-theta,r,c='red',s = 5)
    ax.scatter(np.pi - theta,r,c='green',s = 5)
    ax.scatter(theta-np.pi, r, c= 'purple',s = 5)
    ax.set_rmax(1.2)
    ax.set_title('Train on v1,v2, test on v3', va='bottom')

    plt.savefig('./vis/UCLA/dictionary/TestV1.png')
    #plt.show()
    plt.clf()

# def weightPoles(c_array, r, theta, dictionary, pixnum, err, savedir,name):
def weightPoles(c_array, r, theta, dictionary):
    # c_array = c_array.cpu().numpy()
    # r = r.cpu().numpy()
    # theta = Dtheta.cpu().numpy()
    # dictionary = dictionary.cpu().numpy()
    # imgPath = os.path.join(savedir,'vispoles','seperate')
    # if not os.path.exists(imgPath):
    #     os.makedirs(imgPath)

    lenC = 160 # length of coefficients vector
    uv = 0 # you can choose either u(0) or v(1)
    quad_lik = np.zeros(lenC// 4)
    quad_mean = np.zeros(lenC // 4)
    quad_arr = np.zeros((4, (lenC// 4)))
    # print(‘mu'',c_array, ‘covariance’, err)
    # xaxis = np.linspace(1,161,num=161)
    # plt.stem(xaxis, err)
    # plt.stem(xaxis,c_array.squeeze(),linefmt='C1-',markerfmt='C1o')
    # plt.legend(('cov','mu'))
    # plt.savefig(os.path.join(imgPath,‘err_cov_‘+name+str(pixnum)))
    # plt.close()
    # c_array = np.load(os.path.join(saveDir, ‘carray_%04d’ % vv+‘.npy’)) # Load coefficients saved in test.py
    c_array = c_array[np.newaxis,:,:,:]
    ini = 0 # First frame
    end = 1 # Last frame
    # Note: in this area you will have to choose which pixel location and sequences you want to plot.
    # note: If you need multiple sequences create an outer loop
    for n in range(lenC):
        # check if 4 consecutive values are similar
        # you also have to specify the pixel value (in this case 10290)
        if (n < 40): #(n < 41 and n != 0):
            quad = [np.sum(c_array[ini:end, uv, n+1, 0]), np.sum(c_array[ini:end, uv, n + 41, 0]),
                    np.sum(c_array[ini:end, uv, n + 81, 0]), np.sum(c_array[ini:end, uv, n + 121, 0])]
            quad = np.absolute(quad)
            quad_mean[n - 10] = np.mean(quad)
            quad_lik[n - 10] = np.amax(quad) / quad_mean[n - 10]
            quad_arr[:, n - 10] = quad
    # load parameters
    # Dparams = np.load(os.path.join(saveDir, ‘rr_th_%04d’%vv + ‘.npy’))
    # r = Dparams[0]
    # theta = Dparams[1]
    # load dictionary --> check it’s saved in numpy format
    # dictionary = np.load(os.path.join(saveDir, ‘dictionary_%04d’%vv + ‘.npy’))
    # pole_arr = dictionary[1, 0:lenC].reshape(4, int(lenC / 4))
    pole_arr = dictionary[1, 1:].reshape(4, int(lenC / 4))
    dic_values = np.zeros(pole_arr.shape)
    # compute weights of each pole
    for n in range(40):
        if pole_arr[0, n] > 0:
            dic_values[[0, 1], n] += [quad_arr[0, n], quad_arr[0, n]]
        elif pole_arr[0, n] < 0:
            dic_values[[2, 3], n] += [quad_arr[0, n], quad_arr[0, n]]
        if pole_arr[1, n] > 0:
            dic_values[[0, 3], n] += [quad_arr[1, n], quad_arr[1, n]]
        elif pole_arr[1, n] < 0:
            dic_values[[1, 2], n] += [quad_arr[1, n], quad_arr[1, n]]
        if pole_arr[2, n] < 0:
            dic_values[[0, 1], n] += [quad_arr[2, n], quad_arr[2, n]]
        elif pole_arr[2, n] > 0:
            dic_values[[2, 3], n] += [quad_arr[2, n], quad_arr[2, n]]
        if pole_arr[3, n] < 0:
            dic_values[[0, 3], n] += [quad_arr[3, n], quad_arr[3, n]]
        elif pole_arr[3, n] > 0:
            dic_values[[1, 2], n] += [quad_arr[3, n], quad_arr[3, n]]
    dic_values = (dic_values / np.amax(dic_values)) * 100
    # dic_values[dic_values<85]=0
    for iii in range(1):
        # polar plot
        cm = plt.cm.get_cmap('spring')#(‘Blues’)
        ax = plt.subplot(1, 1, 1, projection='polar')
        ax.scatter(0, 1, c='black')
        ax.scatter(theta, r, s=dic_values[0, :], c=dic_values[0, :], alpha=0.8, edgecolors='none', cmap=cm)
        ax.scatter(-theta, r, s=dic_values[1, :], c=dic_values[1, :], alpha=0.8, edgecolors='none', cmap=cm)
        ax.scatter(np.pi - theta, r, s=dic_values[2, :], c=dic_values[2, :], alpha=0.8, edgecolors='none', cmap=cm)
        ax.scatter(theta - np.pi, r, s=dic_values[3, :], c=dic_values[3, :], alpha=0.8, edgecolors='none', cmap=cm)
        #
        # ax.set_rmax(1.2)
        # ax.errorbar(0, 1, xerr=0, yerr=err[0]*40, fmt=' ',ecolor='b',capsize=0.5,elinewidth=0.2)
        # ax.errorbar(theta, r, xerr=0, yerr=err[1:41]*40,fmt=' ',ecolor='b',capsize=0.5,elinewidth=0.2)
        # ax.errorbar(-theta, r, xerr=0, yerr=err[41:81]*40,fmt=' ',ecolor='b',capsize=0.5,elinewidth=0.2)
        # ax.errorbar(np.pi - theta, r, xerr=0, yerr=err[81:121]*40,fmt=' ',ecolor='b',capsize=0.5,elinewidth=0.2)
        # ax.errorbar(theta - np.pi, r, xerr=0, yerr=err[121:161]*40,fmt=' ',ecolor='b',capsize=0.5,elinewidth=0.2)
        #result = plt.gcf()
        # ax.legend()
        # ax.set_title(“Weighted Poles”)  # , va=‘bottom’
        ax.set_title('WeightedPoles')
        plt.savefig('./weight_pole_testV1', dpi=400)
        plt.close()
        # plt.savefig(‘try’,dpi =400)
        # plt.savefig(‘weight_pole_‘+str(pixnum),dpi =400)
        # plt.savefig(os.path.join(imgPath,'weight_pole_wCov_' +name+str(pixnum)),dpi =400)
        # plt.close()
        #try_image = plt.show()
        #return result


data = scipy.io.loadmat('TestV1_m2.mat')
coeff = data['coeff']
Drr = data['r']
Dtheta = data['theta']
dict = data['dict']

weightPoles(coeff, Drr, Dtheta, dict)

print('check')



#
# N = 4*40
# gpu_id = 1
# T = 36
# dataset = 'NUCLA'
# num_class = 10
# path_list = f"/data/Dan/N-UCLA_MA_3D/lists"
# clip = 'Multi'
# modelRoot = '/home/yuexi/Documents/ModelFile/crossViewModel/'
# modelPath = modelRoot + dataset + '/2Stream/multiClip_lam1051_4LayerGN/'
# stateDict = torch.load(os.path.join(modelPath, '80.pth'))['state_dict']
# Drr = stateDict['dynamicsClassifier.sparsecoding.l1.rr']
# Dtheta = stateDict['dynamicsClassifier.sparsecoding.l1.theta']
#
# 'CV:'
# testSet = NUCLA_viewProjection(root_list=path_list, dataType='2D', clip=clip, phase='test', cam='3,2', T=T,
#                                   target_view='view_2',
#                                   project_view='view_3', test_view='view_1')
# testloader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4)
#
#
# kinetics_pretrain = './pretrained/i3d_kinetics.pth'
# net = twoStreamClassification(num_class=10, Npole=(N+1), num_binary=(N+1), Drr=Drr, Dtheta=Dtheta,
#                                   PRE=0, dim=2, gpu_id=gpu_id, dataType='2D', kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
# net.load_state_dict(stateDict)
# #
# model = net.dynamicsClassifier.sparsecoding
# # plotpoles(r, theta)
# with torch.no_grad():
#     for s, sample in enumerate(testloader):
#
#         inputSkeleton = sample['input_skeleton'].float().cuda(gpu_id)
#         t = inputSkeleton.shape[2]
#         if sample['action'] == 0:
#             for i in range(0, 1):
#                 input_clip = inputSkeleton[:, i, :, :, :].reshape(1, t, -1)
#
#
#                 coeff, dict = model.l1(input_clip, t)
#
#                 # print(coeff.shape, dict.shape)
#                 data = {'coeff':coeff.cpu().numpy(), 'dict':dict.cpu().numpy(), 'r':Drr.cpu().numpy(), 'theta':Dtheta.cpu().numpy()}
#                 scipy.io.savemat('TestV2_m2.mat', mdict=data)
#             # weightPoles(coeff, Drr, Dtheta, dict)
#
#
#
#             print('check')
#
