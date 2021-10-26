############################# Import Section #################################

## Imports related to PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
# from modelZoo.actHeat import imageFeatureExtractor
import torch
from math import sqrt
import numpy as np
import pdb

############################# Import Section #################################

# Create Dictionary
def creatRealDictionary(T,Drr,Dtheta,gpu_id):
    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    # Wones  = Variable(Wones,requires_grad=False)
    # Wones = torch.ones(1)
    for i in range(0,T):
        W1 = torch.mul(torch.pow(Drr,i) , torch.cos(i * Dtheta))
        W2 = torch.mul ( torch.pow(-Drr,i) , torch.cos(i * Dtheta) )
        W3 = torch.mul ( torch.pow(Drr,i) , torch.sin(i *Dtheta) )
        W4 = torch.mul ( torch.pow(-Drr,i) , torch.sin(i*Dtheta) )
        W = torch.cat((Wones,W1,W2,W3,W4),0)

        WVar.append(W.view(1,-1))
    dic = torch.cat((WVar),0)
    # G = torch.norm(dic,p=2,dim=0)
    # # idx = (G == 0).nonzero()
    # idx = G==0
    # nG = G.clone()
    # # print(type(T))
    # nG[idx] = np.sqrt(T)
    # G = nG

    # dic = dic/G

    return dic

def fista(D, Y, lambd,maxIter,gpu_id):
    # D = D.type(dtype=torch.double)
    DtD = torch.matmul(torch.t(D),D)
    L = torch.norm(DtD,2)
    linv = 1/L
    DtY = torch.matmul(torch.t(D),Y)
    # x_old = Variable(torch.zeros(DtD.shape[1],DtY.shape[1]).cuda(gpu_id), requires_grad=True) #can change this to 1? related to x being 2d?
    x_old = Variable(torch.zeros(DtD.shape[1], DtY.shape[2]).cuda(gpu_id), requires_grad=True)  # batch-wised
    t = 1
    y_old = x_old
    # y_old = y_old.type(dtype=torch.double)
    # print('D', D)
    lambd = lambd*(linv.data.cpu().numpy())
    A = Variable(torch.eye(DtD.shape[1]).cuda(gpu_id),requires_grad=True) - torch.mul(DtD,linv)

    DtY = torch.mul(DtY,linv)

    Softshrink = nn.Softshrink(lambd )
    # print('lambd:', lambd)
    with torch.no_grad():
        for ii in range(maxIter):
            Ay = torch.matmul(A,y_old)
            del y_old
            with torch.enable_grad():
                x_new = Softshrink((Ay + DtY))
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
            tt = (t-1)/t_new
            y_old = torch.mul( x_new,(1 + tt))
            y_old -= torch.mul(x_old , tt)
            if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-4:
                x_old = x_new
                break
            t = t_new
            x_old = x_new
            del x_new
    return x_old



class Encoder(nn.Module):
    def __init__(self, Drr, Dtheta, gpu_id):
        super(Encoder, self).__init__()

        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        #self.T = T
        self.gid = gpu_id

    def forward(self,x,T):
        dic = creatRealDictionary(T,self.rr,self.theta,self.gid)

        sparsecode = fista(dic,x,0.1,100,self.gid)

        return Variable(sparsecode), dic

class Decoder(nn.Module):
    def __init__(self,rr,theta, PRE, gpu_id ):
        super(Decoder,self).__init__()

        self.rr = rr
        self.theta = theta
        #self.T = T
        self.PRE = PRE
        self.gid = gpu_id

    def forward(self,x,T):
        dic = creatRealDictionary(T+self.PRE,self.rr,self.theta,self.gid)
        dic = dic.type(dtype=torch.double)
        result = torch.matmul(dic,x)
        return result


class sparseCodingGenerator(nn.Module):
    def __init__(self, Drr, Dtheta, PRE, gpu_id):
        super(sparseCodingGenerator, self).__init__()
        self.l1 = Encoder(Drr, Dtheta, gpu_id)
        self.l2 = Decoder(self.l1.rr,self.l1.theta, PRE, gpu_id)

    def forward(self,x,T):
        coeff, dict = self.l1(x, T)
        return self.l2(coeff,T)

    def forward2(self,x,T):
        return self.l1(x,T)

"""""
class imgFeatureDYAN(nn.Module):
    def __init__(self, Drr, Dtheta, PRE, backbone, gpu_id):
        super(imgFeatureDYAN, self).__init__()
        self.featureExtractor = imageFeatureExtractor(backbone)
        self.sparseCoding = sparseCodingGenerator(Drr, Dtheta, PRE, gpu_id)

    def forward(self,x, T):
        reducedFeature, _ = self.featureExtractor(x)

        inputImgFeat = reducedFeature.view(T, -1).unsqueeze(0)

        outFeature = self.sparseCoding(inputImgFeat, T)

        return outFeature
"""""

def fista_new(D, Y, lambd,maxIter, gpu_id):
    DtD = torch.matmul(torch.t(D),D)
    L = torch.norm(DtD,2)
    linv = 1/L
    DtY = torch.matmul(torch.t(D),Y)
    x_old = torch.zeros(DtD.shape[1],DtY.shape[2]).cuda(gpu_id)
    t = 1
    y_old = x_old
    lambd = lambd*(linv.data.cpu().numpy())
    # print('lambda:', lambd, 'linv:',1/L, 'DtD:',DtD, 'L', L )
    # print('dictionary:', D)
    A = torch.eye(DtD.shape[1]).cuda(gpu_id) - torch.mul(DtD,linv)
    DtY = torch.mul(DtY,linv)
    Softshrink = nn.Softshrink(lambd)
    for ii in range(maxIter):
        # print('iter:',ii, lambd)
        Ay = torch.matmul(A,y_old)
        del y_old
        x_new = Softshrink((Ay + DtY))

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt)
        # pdb.set_trace()
        if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-5:
            x_old = x_new
            # print('Iter:', ii)
            break
        t = t_new
        x_old = x_new
        del x_new
    return x_old

def fista_reweighted(D, Y, lambd, w, maxIter,gpu_id):
    
    DtD = torch.matmul(torch.t(D), D)
    DtY = torch.matmul(torch.t(D), Y)
    # eig, v = torch.eig(DtD, eigenvectors=True)
    # eig, v = torch.linalg.eig(DtD)
    # L = torch.max(eig)
    L = torch.norm(DtD, 2)
    Linv = 1/L
    weightedLambd = (w*lambd) * Linv.data.item()
    x_old = torch.zeros(DtD.shape[1], DtY.shape[2]).cuda(gpu_id)
    # x_old = x_init
    y_old = x_old
    A = torch.eye(DtD.shape[1]).cuda(gpu_id) - torch.mul(DtD,Linv)
    t_old = 1

    const_xminus = torch.mul(DtY, Linv) - weightedLambd.cuda(gpu_id)
    const_xplus = torch.mul(DtY, Linv) + weightedLambd.cuda(gpu_id)

    iter = 0

    while iter < maxIter:
        iter +=1
        Ay = torch.matmul(A, y_old)
        x_newminus = Ay + const_xminus
        x_newplus = Ay + const_xplus
        x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                torch.min(torch.zeros_like(x_newplus), x_newplus)

        t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

        tt = (t_old-1)/t_new
        tt = torch.tensor(tt).cuda(gpu_id)
        y_new = x_new + torch.mul(tt, (x_new-x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-8:
            x_old = x_new
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new

    return x_old


def generateD(rr, theta, row, T):
  W = torch.FloatTensor()
  if isinstance(row, torch.Tensor) and len(row.size()) > 1:
    W1 = torch.mul(torch.pow(rr.unsqueeze(0), torch.t(row)),
            torch.cos(torch.t(row) * theta.unsqueeze(0)))
    W2 = torch.mul(torch.pow(rr.unsqueeze(0), torch.t(row)),
            torch.sin(torch.t(row) * theta.unsqueeze(0)))
    W = torch.stack((W1, W2), 1)
    W = W.view(row.shape[1], -1)
  else:
    W1 = torch.mul(torch.pow(rr, row), torch.cos(row * theta))
    W2 = torch.mul(torch.pow(rr, row), torch.sin(row * theta))
    W = torch.cat((W1, W2), 0)
    W = W.view(1, -1)
  # print('new dictionary has shape:',W.shape)
  return W

class DyanEncoder(nn.Module):
    def __init__(self, Drr, Dtheta, lam, gpu_id):
        super(DyanEncoder, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        # self.T = T
        self.lam = lam
        self.gpu_id = gpu_id

    def forward(self, x,T):
        dic = creatRealDictionary(T, self.rr,self.theta, self.gpu_id)
        # print('rr:', self.rr, 'theta:', self.theta)
        # sparsecode = fista_new(dic,x,self.lam, 200,self.gpu_id)
        i = 0
        w_init = torch.ones(1, dic.shape[1], x.shape[2])
        while i < 3:
            temp = fista_reweighted(dic, x, self.lam, w_init, 200, self.gpu_id)
            w = 1 / (torch.abs(temp) + 1e-2)
            w_init = w/torch.norm(w)
            final = temp
            del temp
            i += 1
        sparseCode = final

        # reconst = torch.matmul(dic, sparsecode)
        return sparseCode, dic

if __name__ == '__main__':
    N = 80
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    gpu_id = 1
    T = 36
    model = DyanEncoder(Drr, Dtheta, lam=1, gpu_id=gpu_id).cuda(gpu_id)

    input = torch.randn(1, T,25*2).cuda(gpu_id)

    out,_ = model(input, T)

    print('check')
