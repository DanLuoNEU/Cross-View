import torch.nn as nn
import scipy.io
import pdb
import torch
import numpy as np
def fista_new(D, Y, lambd, maxIter):
    DtD = torch.matmul(torch.t(D),D)
    L = torch.norm(DtD, 2)
    # L = torch.linalg.matrix_norm(DtD, 2)
    linv = 1/L
    DtY = torch.matmul(torch.t(D),Y)
    x_old = torch.zeros(DtD.shape[1],DtY.shape[2])
    t = 1
    y_old = x_old
    lambd = lambd*(linv.data.cpu().numpy())   # ---> (w*lambd)/L
    # print('lambda:', lambd, 'linv:',1/L, 'DtD:',DtD, 'L', L )
    # print('dictionary:', D)
    A = torch.eye(DtD.shape[1]) - torch.mul(DtD,linv)
    DtY = torch.mul(DtY,linv)
    Softshrink = nn.Softshrink(lambd)

    for ii in range(maxIter):
        # print('iter:',ii, lambd)
        Ay = torch.matmul(A,y_old)
        del y_old
        # temp = w * (Ay + DtY)
        # x_new = Softshrink(temp)
        x_new = Softshrink(Ay+DtY)   # --> shrink(y_k)
        # x_new = lambd * (Ay+DtY)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt)
        # pdb.set_trace()
        if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-8:
            x_old = x_new
            print('Iter:', ii)
            break
        t = t_new
        x_old = x_new
        del x_new
    return x_old


def fista_reweighted(D, Y, lambd, w, maxIter):
    DtD = torch.matmul(torch.t(D), D)
    DtY = torch.matmul(torch.t(D), Y)
    # eig,v = torch.eig(DtD, eigenvectors=True)
    # L = torch.max(eig)
    # L = torch.linalg.matrix_norm(DtD, 2)
    # L = torch.linalg.matrix_norm(D,2)** 2
    L = torch.abs(torch.linalg.eigvals(DtD)).max()
    Linv = 1/L
    lambd = (w*lambd) * Linv.data.item()
    x_old = torch.zeros(DtD.shape[1], DtY.shape[2])
    # x_old = x_init
    y_old = x_old
    A = torch.eye(DtD.shape[1]) - torch.mul(DtD,Linv)
    t_old = 1

    const_xminus = torch.mul(DtY, Linv) - lambd
    const_xplus = torch.mul(DtY, Linv) + lambd

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
        y_new = x_new + torch.mul(tt, (x_new-x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
        if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-8:
            x_old = x_new
            # print(iter)
            break

        t_old = t_new
        x_old = x_new
        y_old = y_new
    print(iter)

    return x_old



# data = scipy.io.loadmat('../synthetic_gumbel.mat')

dataPath = '/data/Yuexi/Cross_view/1115/binaryCode/coeff_bi_Y_v2_action05.mat'
dictionary = scipy.io.loadmat('/home/dan/ws/2021-CrossView/matfiles/1115_coeff-wi-bin/Dictionary_clean_20211116.mat')




# dictionary = torch.tensor(data['dictionary'])
# coeff = torch.tensor(data['coeff'])

# Y = torch.matmul(dictionary, coeff)

# pdb.set_trace()

# y = Y[10].unsqueeze(0)

w_init = torch.ones(1,61,1)
# c_init = torch.zeros(61,1)
lambd = 1
# c_sparse = fista_new(dictionary, y, lambd, 200 )

i = 0
while i < 3:

    c_sparse_re = fista_reweighted(dictionary, y,lambd, w_init, 200 )


    w = 1/(torch.abs(c_sparse_re) + 1e-2)
    w_norm = w/torch.norm(w,2)
    # print('iter:',i, w[0].t())
    w_init = w_norm * 61
    # w_init = w
        # lam_int = lam
    i+=1
    # c_init = c_sparse_re

    c_final = c_sparse_re
    del c_sparse_re

print('check')