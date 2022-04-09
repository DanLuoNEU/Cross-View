import os
import math
import random
import numpy as np
from math import isclose
import matplotlib.pyplot as plt
import torch
import torch.nn
from torch.autograd import Variable
from modelZoo.DyanOF import OFModel, fista
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def weightPoles(c_array, Drr, Dtheta, dictionary):
    c_array = c_array.cpu().numpy()
    r = Drr.cpu().numpy()
    theta = Dtheta.cpu().numpy()
    dictionary = dictionary.cpu().numpy()
    lenC = 160  # length of coefficients vector
    uv = 0  # you can choose either u(0) or v(1)
    quad_lik = np.zeros(lenC // 4)
    quad_mean = np.zeros(lenC // 4)
    quad_arr = np.zeros((4, (lenC // 4)))
    c_array = c_array[np.newaxis, :, :, :]
    ini = 0  # First frame
    end = 1  # Last frame
    # Note: in this area you will have to choose which pixel location and sequences you want to plot.
    # note: If you need multiple sequences create an outer loop
    for n in range(lenC):
        print(n)
        # check if 4 consecutive values are similar
        # you also have to specify the pixel value (in this case 10290)
        if (n < 40):  # (n < 41 and n != 0):
            quad = [np.sum(c_array[ini:end, uv, n + 1, 0]), np.sum(c_array[ini:end, uv, n + 41, 0]),
                    np.sum(c_array[ini:end, uv, n + 81, 0]), np.sum(c_array[ini:end, uv, n + 121, 0])]
            quad = np.absolute(quad)
            quad_mean[n - 10] = np.mean(quad)
            quad_lik[n - 10] = np.amax(quad) / quad_mean[n - 10]
            quad_arr[:, n - 10] = quad
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
        for iii in range(40):
            # polar plot
            cm = plt.cm.get_cmap('spring')  # (‘Blues’)
            ax = plt.subplot(1, 1, 1, projection='polar')
            ax.scatter(0, 1, c='black')
            ax.scatter(theta, r, s=dic_values[0, :], c=dic_values[0, :], alpha=0.8, edgecolors='none', cmap=cm)
            ax.scatter(-theta, r, s=dic_values[1, :], c=dic_values[1, :], alpha=0.8, edgecolors='none', cmap=cm)
            ax.scatter(np.pi - theta, r, s=dic_values[2, :], c=dic_values[2, :], alpha=0.8, edgecolors='none', cmap=cm)
            ax.scatter(theta - np.pi, r, s=dic_values[3, :], c=dic_values[3, :], alpha=0.8, edgecolors='none', cmap=cm)
            ax.set_title('WeightedPoles')
            plt.savefig('./vis/UCLA/dictionary/weight_pole_testV3', dpi = 400)
            plt.close()
def gridRing(N):
    # epsilon_low = 0.25
    # epsilon_high = 0.15
    # rmin = (1 - epsilon_low)
    # rmax = (1 + epsilon_high)

    # epsilon_low = 0.25
    epsilon_low = 0.15
    epsilon_high = 0.15
    rmin = (1 - epsilon_low)
    rmax = (1 + epsilon_high)

    thetaMin = 0.001
    thetaMax = np.pi - 0.001
    delta = 0.001
    # Npole = int(N / 4)
    Npole = int(N/2)
    Pool = generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax)
    M = len(Pool)

    idx = random.sample(range(0, M), Npole)
    P = Pool[idx]
    # Pall = np.concatenate((P, -P, np.conjugate(P), np.conjugate(-P)), axis=0)
    Pall = np.concatenate((P, np.conjugate(P)), axis=0)  # mirror once

    return P, Pall


## Generate the grid on poles
def generateGridPoles(delta, rmin, rmax, thetaMin, thetaMax):
    rmin2 = pow(rmin, 2)
    rmax2 = pow(rmax, 2)
    xv = np.arange(-rmax, rmax, delta)
    x, y = np.meshgrid(xv, xv, sparse=False)
    mask = np.logical_and(np.logical_and(x ** 2 + y ** 2 >= rmin2, x ** 2 + y ** 2 <= rmax2),
                          np.logical_and(np.angle(x + 1j * y) >= thetaMin, np.angle(x + 1j * y) <= thetaMax))
    px = x[mask]
    py = y[mask]
    P = px + 1j * py

    return P


def get_Drr_Dtheta(N):
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    return Drr, Dtheta

def getRowSparsity(inputDict):
    rowNum = inputDict.shape[0]
    L = inputDict.shape[1]
    count = 0
    for i in range(0, rowNum):
        dictRow = inputDict[i,:].unsqueeze(0)
        if len(dictRow.nonzero()) <= round(0.6*L):
            count+=1
        else:
            continue
    rowSparsity = count
    return rowSparsity

def set_to_zeros(input):
    # thresh = 0.0000
    for i in range(input.shape[0]):
        if isclose(input[i], 0, abs_tol=1e-3):
            input[i] = 0
    return input

def get_normalized_data(inputData, type):
    if type == 'Gaussian':
        mean = torch.mean(inputData)
        std = torch.std(inputData)
        outputData = (inputData - mean)/std


    elif type == '01':
        Max = torch.max(inputData)
        Min = torch.min(inputData)
        outputData = (inputData - Min)/Max

    return outputData


def getOneDimData(input, dim):
    'input = 3d tensor'
    'dim = skeleton dim'
    L = np.arange(0, input.shape[-1], dim)

    output = input[:, :, L]

    return output

def plotting(input, reconstruction,key_frame, imageName, saveDir, seqNum):
    if seqNum > 1:

        seq1 = input[:, seqNum]
        seq2 = reconstruction[:, seqNum]
        y_key = -5 * np.ones(seq1.shape)
    else:
        seq1 = input
        seq2 = reconstruction
        y_key = -5 * np.ones(seq1.shape)

    y_key[key_frame,:] = seq1[key_frame,:]
    T = np.arange(0, input.shape[0],1)
    # plt.plot(T, seq1, 'b', T, seq2, 'r', T, y_key,'g*')
    plt.plot(T, seq1, 'b', label='gt')
    plt.plot(T, seq2, 'r', label='recover')
    plt.plot(T, y_key, 'g*', label='key frames')

    plt.legend()
    plt.title(imageName)
    plt.savefig(os.path.join(saveDir, imageName + '.png'))


def loadModel(ckpt_file, T, gpu_id):
    loadedcheckpoint = torch.load(ckpt_file, map_location=lambda storage, location: storage)
    #loadedcheckpoint = torch.load(ckpt_file)
    stateDict = loadedcheckpoint['state_dict']

    # load parameters
    Dtheta = stateDict['l1.theta']
    Drr    = stateDict['l1.rr']
    model = OFModel(Drr, Dtheta, T, gpu_id)
    model.cuda(gpu_id)

    return model

def get_Dictionary(T, numPole, gpu_id, addOne):

    P, Pall = gridRing(numPole)
    # Drr = np.zeros(1)
    # Dtheta = np.zeros(1)
    # P = 0.625 + 1j * 0.773
    # print(P)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float().cuda(gpu_id)
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float().cuda(gpu_id)

    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    Wones = Variable(Wones, requires_grad=False)
    for i in range(0, T):
        W1 = torch.mul(torch.pow(Drr, i), torch.cos(i * Dtheta))

        W3 = torch.mul(torch.pow(Drr, i), torch.sin(i * Dtheta))
        if addOne:
            W = torch.cat((Wones, W1, W3), 0)
        else:
            W = torch.cat((W1, W3), 0)
        # W = torch.cat((W1, W3), 0)
        WVar.append(W.view(1, -1))
    dic = torch.cat((WVar), 0)
    G = torch.norm(dic, p=2, dim=0)
    idx = (G == 0).nonzero()
    nG = G.clone()
    nG[idx] = np.sqrt(T)
    G = nG

    dic = dic / G

    return dic

def getNoneZeroDict(Dictionary, Coefficience, seq_num, single):
    if single:
        c1 = Coefficience[:, seq_num]
        c1 = set_to_zeros(c1)

        nonzero_idx = torch.nonzero(c1)
        # print('coeff:', c1.shape, 'nonzero_idx:', nonzero_idx)
        C1_nonzero = c1[nonzero_idx]  # [Nx1]

        # print(nonzero_idx)
        # print(dictionary.shape)
        Dict_nonzero = Dictionary[:, nonzero_idx].squeeze(2)
        # Y = torch.matmul(Dict_nonzero, C1_nonzero)

    else:
        List = torch.nonzero(Coefficience)
        nonzero_list = torch.unique(List[:, 0])

        Dict_nonzero = Dictionary[:, nonzero_list]

    return Dict_nonzero, C1_nonzero

def get_recover(D, y, key_set):
    D_r = D[key_set, :]
    if len(y.shape) == 3:
        y_r = y[:,key_set, :]
    else:
        y_r = y[key_set,:]

    dtd_r = torch.matmul(D_r, D_r.t())

    a = torch.matmul(D_r.t(), torch.inverse(dtd_r))
    coef_r = torch.matmul(a, y_r)

    y_hat = torch.matmul(D, coef_r)

    return y_hat,coef_r

def get_recover_fista(D, y, key_set, gpu_id):
    if type(D) is np.ndarray:
        D = torch.Tensor(D)

    D_r = D[key_set]
    # print(y.shape, 'is cuda:', y.is_cuda, 'key_set:', key_set)
    if len(y.shape)==3:
        y_r = y[:,key_set]
    else:
        y_r = y[key_set]


    if D.is_cuda:
        c_r = fista(D_r, y_r, 0.1, 100, gpu_id)  # lambda=0.4 is the best one
        y_hat = torch.matmul(D, c_r)
    else:
        c_r = fista(D_r.cuda(gpu_id), y_r, 0.1, 100, gpu_id)
        y_hat = torch.matmul(D.cuda(gpu_id), c_r)

    return y_hat,c_r

def get_Cr_CoCr(D, y, key_set):
    D_r = D[key_set, :]
    if len(y.shape) == 3:
        y_r = y[:, key_set, :]
    else:
        y_r = y[key_set, :]

    dtd_r = torch.matmul(D_r, D_r.t())

    a = torch.matmul(D_r.t(), torch.inverse(dtd_r))
    coef_r = torch.matmul(a, y_r).squeeze(0).cpu().numpy()

    cov_cr = np.matmul(np.matmul(coef_r.transpose(),np.cov(coef_r)), coef_r)

    return coef_r, cov_cr




def DrawGaussian(img, pt, sigma):
    tmpSize = int(np.math.ceil(3 * sigma))
    # if math.isnan(float(pt[0] - tmpSize)):
    #     print('NaN')
    ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
    br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]

    if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
        return img

    size = 2 * tmpSize + 1
    g = Gaussian(size)

    g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
    g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]

    img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
    img_y = [max(0, ul[1]), min(br[1], img.shape[0])]

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def Gaussian(sigma):
  if sigma == 7:
    return np.array([0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.2301,  0.5205,  0.8494,  1.0000,  0.8494,  0.5205,  0.2301,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529]).reshape(7, 7)
  elif sigma == n:
    return g_inp
  else:
    raise Exception('Gaussian {} Not Implement'.format(sigma))



# class KroneckerProduct(nn.Module):
#     """
#     Class to perform the Kronecker Product inside the Autograd framework.
#     See: https://en.wikipedia.org/wiki/Kronecker_product
#     Computes Kronecker Product for matrices A and B, where
#     A has shape (batch_size, Ar, Ac) and B has shape (batch_size, Br, Bc)
#     Usage:
#       * Initialise an instance of this class by specifying the shapes of A and B.
#       * Call the class on A and B, which calls the forward function.
#     """
#
#     def __init__(self, A_shape, B_shape):
#         """
#         Inputs:
#             A_shape         A tuple of length 2 specifying the shape of A---(Ar, Ac)
#             B_shape         A tuple of length 2 specifying the shape of B---(Br, Bc)
#         """
#
#         super(KroneckerProduct, self).__init__()
#
#         # Extract rows and columns.
#         Ar, Ac = A_shape
#         Br, Bc = B_shape
#
#         # Output size.
#         Fr, Fc = Ar * Br, Ac * Bc
#
#         # Shape for the left-multiplication matrix
#         left_mat_shape = (Fr, Ar)
#         # Shape for the right-multiplication matrix.
#         right_mat_shape = (Ac, Fc)
#
#         # Identity matrices for left and right matrices.
#         left_eye = torch.eye(Ar)
#         right_eye = torch.eye(Ac)
#
#         # Create left and right multiplication matrices.
#         self.register_buffer('left_mat', torch.cat([x.view(1, -1).repeat(Br, 1) for x in left_eye], dim=0))
#         self.register_buffer('right_mat', torch.cat([x.view(-1, 1).repeat(1, Bc) for x in right_eye], dim=1))
#
#         # Unsqueeze the batch dimension.
#         self.left_mat = self.left_mat.unsqueeze(0)
#         self.right_mat = self.right_mat.unsqueeze(0)
#
#         # Function to expand A as required by the Kronecker Product.
#         self.A_expander = lambda A: torch.bmm(self.left_mat.expand(A.size(0), Fr, Ar),
#                                               torch.bmm(A, self.right_mat.expand(A.size(0), Ac, Fc)))
#
#         # Function to tile B as required by the kronecker product.
#         self.B_tiler = lambda B: B.repeat(1, Ar, Ac)
#
#     def forward(self, A, B):
#         """
#         Compute the Kronecker product for A and B.
#         """
#         # This operation is a simple elementwise-multiplication of the expanded and tiled matrices.
#         return self.A_expander(A) * self.B_tiler(B)



def get_reducedDictionary(Drr, Dtheta, THD_distance):
    # Drr_Dict = mat_Dict_DYAN['Drr'] # 1x80
    # Dtheta_Dict = mat_Dict_DYAN['Dtheta'] # 1x80
    order_theta = np.argsort(np.insert(np.squeeze(Dtheta),0,0))
    Drr_temp = np.insert(np.squeeze(Drr),0,1)
    Dtheta_temp = np.insert(np.squeeze(Dtheta),0,0)
    list_poles_far=[(1,0)]
    r_reduced = []
    theta_reduced = []
    for i_theta in order_theta:
        r_temp = Drr_temp[i_theta]
        theta_temp = Dtheta_temp[i_theta]
        bool_close=False
        for r_far,theta_far in list_poles_far:
            if (distance_polar(r_temp,theta_temp,r_far,theta_far) < THD_distance) \
                or (distance_polar(r_temp,-theta_temp,r_far,theta_far) < THD_distance):
                bool_close=True
        if not bool_close:
            list_poles_far.append((r_temp,theta_temp))
            r_reduced.append(r_temp)
            theta_reduced.append(theta_temp)

    return list_poles_far, r_reduced, theta_reduced
    


def distance_polar(rho_1,theta_1,rho_2,theta_2):
    return np.sqrt(rho_1**2+rho_2**2-2*rho_1*rho_2*(np.cos(theta_1)*np.cos(theta_2)+np.sin(theta_1)*np.sin(theta_2)))

