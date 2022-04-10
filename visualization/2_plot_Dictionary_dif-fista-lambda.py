import os
import sys
from PIL import Image
import numpy as np
from numpy.lib.function_base import _DIMENSION_NAME
import scipy.io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import torch
# import torchvision
# from torch.utils.data import DataLoader
# from modelZoo.BinaryCoding import *
# from dataset.NUCLA_ViewProjection_trans import *

def creat_Real_Dictionary(T,Drr,Dtheta,theta_max_pi=False):
    WVar = []
    Wones = np.ones(1)
    D_rr = np.squeeze(Drr)
    D_theta = np.squeeze(Dtheta)
    # Wones  = Variable(Wones,requires_grad=False)
    # Wones = torch.ones(1)
    for i in range(0,T):
        if not theta_max_pi:
            W1 = np.multiply(np.power( D_rr,i), np.cos(i*D_theta))
            W2 = np.multiply(np.power(-D_rr,i), np.cos(i*D_theta))        
            W3 = np.multiply(np.power( D_rr,i), np.sin(i*D_theta))
            W4 = np.multiply(np.power(-D_rr,i), np.sin(i*D_theta))
            W = np.concatenate((Wones,W1,W2,W3,W4),0)
        else:
            W1 = np.multiply(np.power( D_rr,i), np.cos(i*D_theta))
            W2 = np.multiply(np.power( D_rr,i), np.sin(i*D_theta))        
            W = np.concatenate((Wones,W1,W2),0)

        WVar.append(W.reshape(1,-1))
    dic = np.concatenate((WVar),0)

    return dic

def plotpoles(r, theta):
    # r = model.l1.rr.data.cpu().numpy()          #r should be numpy array with size (20,)
    # theta = model.l1.theta.data.cpu().numpy()   #same as r
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

def weightPoles(path_exp, theta_max_pi=False, num_used_top=30, dim_data_draw=0, 
                weight_complex=False, draw_dict=True, draw_traj=True, draw_pole=True):
    ''' This function can plot poles with visualized weight for different lambda
    Dictionaries Pole Comparison for each file
    One trajectory with poles for 

    Input: 
        theta_max_pi: bool, False means Dictionary in Original DYAN paper, with 2 pair of conjugates for each pole, 0<= theta <= pi/2
                            True means Dictionary in Improved DYAN, with 1 pair of conjugate for each pole, 0 <= theta <= pi
        num_used_top: int,  the number of most used poles to draw
        dim_data_draw:     int,  the dimension of data to draw
    Output:
        figures for weighted poles of each C under path_exp
        
        Suppose .mat file contain:
            C       : Coefficient Matrix,    num_sequences x num_poles x (num_joints x dim), 12 x 1 x 161 x 50 e.g.
            D_r     : Radius in Dictionary,  1 x 40
            D_theta : Theta in Dictionary,   1 x 40
            D       : Dictionary,            T x Num_poles, 21 x 161 e.g.
            NOTE: 
            Original DYAN Dictionary
                D_1: [1, rho_1 (cos theta), rho_2 (cos theta),..., rho_(N) (cos theta),
                         rho_1 (sin theta), rho_2 (sin theta),..., rho_(N) (sin theta),
                        -rho_1 (cos theta),-rho_2 (cos theta),...,-rho_(N) (cos theta),
                        -rho_1 (sin theta),-rho_2 (sin theta),...,-rho_(N) (sin theta)]
                for each pole,
                    weight_c     = |c|      = |C_1-jC_2|
                    weight_c'    = |c'|     = |C_3-jC_4|
                    weight_c_all = |c|+|c'| = |C_1-jC_2| + |C_3-jC_4|
            Improved DYAN Dictionary
                D_1: [1, rho_1 (cos theta), rho_2 (cos theta),..., rho_(N) (cos theta),
                         rho_1 (sin theta), rho_2 (sin theta),..., rho_(N) (sin theta)]
                for each pole,
                    weight_c = |c| = |C_1-jC_2|
        num_used_top: 
        dim_data_draw:
    '''
    # Count Coeff and Dictionary files
    list_name_file = os.listdir(path_exp)
    list_name_mat = []
    for name_file in list_name_file:
        if '.mat' in name_file:
            list_name_mat.append(name_file)
    list_name_mat.sort()

    num_subplots = len(list_name_mat)
    figs_dicts, axs_dicts = plt.subplots(num_subplots//3 if num_subplots%3==0 else num_subplots//3+1, 3, 
                                            subplot_kw=dict(projection='polar'),figsize=(18,18))
    figs_traj, axs_traj = plt.subplots(num_subplots//3 if num_subplots%3==0 else num_subplots//3+1, 3,figsize=(18,18))
    figs_wp, axs_wp = plt.subplots(num_subplots, 3, subplot_kw=dict(projection='polar'), figsize=(18,18))

    list_weight = [] # Used to store data for histogram
    # Read Dictionary file
    for id_subplot, name_mat in enumerate(list_name_mat):
        # Load Coefficient Matrix
        path_mat = os.path.join(path_exp, name_mat)
        data = scipy.io.loadmat(path_mat)
        Coeff_DYAN = data['Coeff']
        Drr_Dict = data['rr']
        Dtheta_Dict = data['theta']
        Dict_Dict = data['dictionary']
        ################### Compute Weights for each pole ############################
        num_sequences, _, num_poles, dim_data = Coeff_DYAN.shape # T,1,161,50 for now
        num_poles_needed = (num_poles-1)//2
        
        # Draw Dictionary
        axs_dicts[id_subplot//3,id_subplot%3].set_title(name_mat.split('.')[0], fontsize=12)
        for i_dict in range(num_poles_needed):
            r_t = Drr_Dict[0, i_dict]
            theta_t = Dtheta_Dict[0, i_dict]
            if num_subplots!=3:
                axs_dicts[id_subplot//3,id_subplot%3].scatter( theta_t, r_t, s=20, c='b', alpha=0.8, edgecolors='none')
                axs_dicts[id_subplot//3,id_subplot%3].scatter(-theta_t, r_t, s=20, c='b', alpha=0.8, edgecolors='none')
            else:
                axs_dicts[id_subplot%3].scatter( theta_t, r_t, s=20, c='b', alpha=0.8, edgecolors='none')
                axs_dicts[id_subplot%3].scatter(-theta_t, r_t, s=20, c='b', alpha=0.8, edgecolors='none')
        
        # Draw trajectories and the most used poles
        ## Draw trajectories
        # Create Dictionary
        Dict_DYAN=creat_Real_Dictionary(36, Drr_Dict, Dtheta_Dict, theta_max_pi=theta_max_pi)
        # Compute all Trajectories for dim_data=dim_data_drwa among 21 frames
        Traj = np.asarray([np.matmul(Dict_DYAN,np.squeeze(sequence_Coeff_DYAN)) for sequence_Coeff_DYAN in Coeff_DYAN])
        if num_subplots!=3:
            axs_traj[id_subplot//3,id_subplot%3].set_title(name_mat.split('.')[0], fontsize=12)
            for t in range(Traj.shape[0]):
                axs_traj[id_subplot//3,id_subplot%3].plot(np.arange(1,37),Traj[t, : , dim_data_draw])
        else:
            axs_traj[id_subplot%3].set_title(name_mat.split('.')[0], fontsize=12)
            for t in range(Traj.shape[0]):
                axs_traj[id_subplot%3].plot(np.arange(1,37),Traj[t, : , dim_data_draw])

        ## Draw mostly used poles
        # # Stupid Bird way
        # weight_poles = np.zeros((num_poles_needed, dim_data))
        # for id_pole in range(num_poles_needed):
        #     weight_C = np.abs(Coeff_DYAN[:,0,1+id_pole*2,] - Coeff_DYAN[:,0,1+id_pole*2+1,]*(0+1j))
        #     weight_poles_p = np.mean(weight_C,axis=0)
        #     weight_poles[id_pole,] = weight_poles_p
        # weight_poles_test = np.mean(np.abs(Coeff_DYAN[:,0,1::2,] - Coeff_DYAN[:,0,2::2,]*(0+1j)),axis=0)
        # print(False in (weight_poles==weight_poles_test))
        if not theta_max_pi:
            # 2 pairs of conjugate poles
            if weight_complex:
                weight_poles_rho_pos = np.mean(np.abs(Coeff_DYAN[:,0,1:1+num_poles_needed//2,] - 
                                                    Coeff_DYAN[:,0,1+num_poles_needed//2:1+num_poles_needed,]*(0+1j)),axis=0)
                weight_poles_rho_neg = np.mean(np.abs(Coeff_DYAN[:,0,1+num_poles_needed:1+num_poles_needed//2*3,] - 
                                                    Coeff_DYAN[:,0,1+num_poles_needed//2*3:,]*(0+1j)),axis=0)
            else:
                weight_poles_rho_pos = np.mean(np.abs(Coeff_DYAN[:,0,1:1+num_poles_needed//2,]) + 
                                               np.abs(Coeff_DYAN[:,0,1+num_poles_needed//2:1+num_poles_needed,]),axis=0)
                weight_poles_rho_neg = np.mean(np.abs(Coeff_DYAN[:,0,1+num_poles_needed:1+num_poles_needed//2*3,]) + 
                                               np.abs(Coeff_DYAN[:,0,1+num_poles_needed//2*3:,]),axis=0)
            weight_poles=np.zeros((num_poles_needed, dim_data))
            weight_poles[0::2]=weight_poles_rho_pos
            weight_poles[1::2]=weight_poles_rho_neg
            weight_poles_rho=weight_poles_rho_neg+weight_poles_rho_pos
        else:
            # 1 pair of conjugate poles
            if weight_complex:
                weight_poles = np.mean(np.abs(Coeff_DYAN[:,0,1:1+num_poles_needed,] - Coeff_DYAN[:,0,1+num_poles_needed:,]*(0+1j)),axis=0)
            else:
                weight_poles = np.mean(np.abs(Coeff_DYAN[:,0,1:1+num_poles_needed,]) + np.abs(Coeff_DYAN[:,0,1+num_poles_needed:,]),axis=0)
        list_weight.append(weight_poles)
        ind_C = np.argsort(weight_poles[:, dim_data_draw])[-num_used_top:]
        # Plot Circle
        axs_wp[id_subplot,0].set_title(name_mat.split('.')[0], fontsize=12)
        
        size_point=weight_poles[ind_C,dim_data_draw]*1000
        if not theta_max_pi:
            ind_C_rho = np.argsort(weight_poles_rho[:, dim_data_draw])[-num_used_top//2:]
            size_point_rho = weight_poles_rho[ind_C_rho, dim_data_draw]*1000//2

        for id_dims_data_weighted_poles in range(3):
            axs_wp[id_subplot,id_dims_data_weighted_poles].scatter(0,1,c='black')
            for id_size,id_C in enumerate(ind_C):
                if not theta_max_pi:
                    # Plot using (rho, theta),(-rho, theta)
                    id_pole = id_C//2
                    r_t = Drr_Dict[0,id_pole]
                    theta_t = Dtheta_Dict[0,id_pole]

                    id_quadrant=id_C%2
                    if id_quadrant == 0:
                        # positive rho, 1 and 4 quadrant
                        axs_wp[id_subplot//3,id_subplot%3].scatter( theta_t, r_t, s=size_point[id_size], c='b', alpha=0.8, edgecolors='none')
                        axs_wp[id_subplot//3,id_subplot%3].scatter(-theta_t, r_t, s=size_point[id_size], c='b', alpha=0.8, edgecolors='none')
                    if id_quadrant == 1:
                        # negative rho, 2 and 3 quadrant
                        axs_wp[id_subplot//3,id_subplot%3].scatter(np.pi - theta_t, r_t, s=size_point[id_size], c='b', alpha=0.8, edgecolors='none')
                        axs_wp[id_subplot//3,id_subplot%3].scatter(theta_t - np.pi, r_t, s=size_point[id_size], c='b', alpha=0.8, edgecolors='none')
                    # Plot using (rho, theta)
                    if id_size < num_used_top//2:
                        id_pole_rho = ind_C_rho
                        r_t_rho = Drr_Dict[0,id_pole_rho]
                        theta_t_rho = Dtheta_Dict[0,id_pole_rho]
                        axs_wp[id_subplot//3,id_subplot%3].scatter( theta_t_rho, r_t_rho, s=size_point_rho[id_size], c='r', alpha=0.8, edgecolors='none')
                        axs_wp[id_subplot//3,id_subplot%3].scatter(-theta_t_rho, r_t_rho, s=size_point_rho[id_size], c='r', alpha=0.8, edgecolors='none')
                else:
                    id_pole = id_C
                    r_t = Drr_Dict[0,id_pole]
                    theta_t = Dtheta_Dict[0,id_pole]
                    
                    axs_wp[id_subplot,id_dims_data_weighted_poles].scatter( theta_t, r_t, s=size_point[id_size], c='b', alpha=0.8, edgecolors='none')
                    axs_wp[id_subplot,id_dims_data_weighted_poles].scatter(-theta_t, r_t, s=size_point[id_size], c='b', alpha=0.8, edgecolors='none')
        
        # plt.savefig(os.path.join(path_exp, f'weight_poles_{num_used_top}.png'), dpi=400)
        ######################  Draw one dim_data Trajectory of each t ########################
        
    if draw_dict:
        figs_dicts.savefig(os.path.join(path_exp, f'dicts_dif_lambda.png'), dpi=400)
    if draw_traj:
        figs_traj.savefig(os.path.join(path_exp, f'trajectories_{num_sequences}_{dim_data_draw}.png'), dpi=400)
    if draw_pole:
        figs_wp.savefig(os.path.join(path_exp, f'poles_weight_{dim_data_draw}_{3}.png'), dpi=400)
        

    # np.column_stack for histogram


def main():
    try:
        path_exp=sys.argv[1]
    except:
        path_exp="/home/dan/ws/2021-CrossView/matfiles/1106_dif-lambda_pi-rho"

    # data = scipy.io.loadmat(path_mat_dic)
    # weightPoles(data['Coeff'], data['Drr'], data['Dtheta'], data['dictionary'],dim_data_draw=0)
    for i in range(5):
        weightPoles(path_exp, theta_max_pi=True, num_used_top=30, dim_data_draw=i, 
                    weight_complex=False, draw_dict=False, draw_traj=True, draw_pole=False)

if __name__ == '__main__':
    main()
