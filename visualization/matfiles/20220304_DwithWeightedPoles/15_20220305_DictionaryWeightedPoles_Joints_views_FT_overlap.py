# 03/04/2022, Dan
# This file is used to draw Dictionary with weighted poles for several joints for setup3, v3 for test
# 03/08/2022, Dan
# Bug: fix Drr_Dict/Dtheta_Dict bug, 
#      Should be Drr_temp and Dtheta_temp, because (r,theta)=(1,0) is not in Drr_Dict/Dtheta_Dict
from io import IOBase
import os
import cv2
import sys
from PIL import Image
import numpy as np
from numpy.core.fromnumeric import argsort
from numpy.lib.function_base import _DIMENSION_NAME
import scipy.io
import pandas as pd
import imageio
import colorsys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

dir_default='/home/dan/ws/2021-CrossView/matfiles/20220304_DwithWeightedPoles/20211129/setup3'

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

THD_distance=0.05
def distance_polar(rho_1,theta_1,rho_2,theta_2):
    return np.sqrt(rho_1**2+rho_2**2-2*rho_1*rho_2*(np.cos(theta_1)*np.cos(theta_2)+np.sin(theta_1)*np.sin(theta_2)))

#38
part_body=['Right Arm', 'Left Arm', 'Right Leg', 'Left Leg','Head and Body', 'Feet']
index_body=[[4,5, 6,7, 8,9], # right arm
            [10,11, 12,13, 14,15], # left arm
            [16,17, 18,19, 20,21, 22,23],# right leg
            [24,25, 26,27, 28,29],# left leg
            [0,1, 2,3, 30,31, 32,33, 34,35, 36,37], #head and body
            [38,39, 40,41, 42,43, 44,45, 46,47, 48,49]] # feet
index_arm_R=[[4,5], [6,7], [8,9]] # shoulder, elbow, hand
index_arm_L=[[10,11], [12,13], [14,15]] # shoulder, elbow, hand
dict_arm={'ShoulderRight':[4,5], 'ElbowRight':[6,7], 'HandRight':[8,9],
            'ShoulderLeft':[10,11], 'ElbowLeft':[12,13], 'HandLeft':[14,15]}

color_views={'v1':'y','v2':'r','v3':'b'}
dict_dim_data={'x':0, 'y':1}

def plot_subplot(mat_data, id_act, name_view, index_joint, id_dim_data, mk, c,
                Drr_Dict, Dtheta_Dict, num_poles_needed, num_used_top,
                axs_dict_temp):
    # if id_act==0: continue
    bi_action_act=mat_data['bi_action'][0,id_act].squeeze()
    num_seqs, num_clips, num_poles, dim_data = bi_action_act.shape

    weight_all_poles=bi_action_act[:,:,:,index_joint]
    weight_avg_poles=np.mean(weight_all_poles, axis=(0,1))
    weight_avg_poles_rho_pos=weight_avg_poles[1:(num_poles_needed+1)]
    weight_avg_poles_rho_neg=weight_avg_poles[num_poles_needed+1:]
    weight_avg_poles_rho=np.zeros((1+num_poles_needed,2))
    weight_avg_poles_rho[0]=weight_avg_poles[0]
    weight_avg_poles_rho[1:]=weight_avg_poles_rho_pos+weight_avg_poles_rho_neg

    ind_poles_rho = np.argsort(weight_avg_poles_rho[:,id_dim_data])[-num_used_top:]
    size_point_rho = weight_avg_poles_rho[ind_poles_rho,id_dim_data]*100//2
    
    min_avg_weight=weight_avg_poles_rho[ind_poles_rho[0],id_dim_data]
    max_avg_weight=weight_avg_poles_rho[ind_poles_rho[-1],id_dim_data]
    
    axs_dict_temp.tick_params(axis='x', labelsize=5)
    axs_dict_temp.tick_params(axis='y', labelsize=5)
    for id_size, id_pole in enumerate(ind_poles_rho):
        r_t = Drr_Dict[id_pole]
        theta_t = Dtheta_Dict[id_pole]
        
        axs_dict_temp.scatter( theta_t, r_t, s=size_point_rho[id_size], marker=mk, c=c, alpha=0.6, edgecolors='none')

    return axs_dict_temp

def weightPoles(path_exp, num_used_top=10, dim_data_draw=0,
                theta_max_pi=True, weight_complex=False,
                draw_weight=False, draw_traj=False, draw_cmap=False, draw_bin=False):
    ''' This function can plot poles with visualized weight
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
    
    HSV(Hue, Saturation, Value) -> Dict_Theta, Dict_R, Binary Code

    '''
    # Count Coeff and Dictionary files
    list_name_mat = os.listdir(path_exp)
    list_name_C_B = []
    for name_file in list_name_mat:
        if '.mat' in name_file:
            if 'coeff_bi_' in name_file:
                list_name_C_B.append(name_file)
            elif 'dictionary' in name_file.lower():
                name_Dict = name_file
    list_name_C_B.sort()
    # If vis folder is not there, make its directory
    dir_save = os.path.join(path_exp,'20220314_0_comp_A1V2V3-A6V2V3-V3A1A6_rThetaFixed')
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    num_subplots = len(list_name_C_B)
    figs_dict_color, axs_dict_color = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
    figs_dict_joint, axs_dict_joint = plt.subplots(2, 3, subplot_kw=dict(projection='polar'))
    figs_dict_joint_cmp, axs_dict_joint_cmp = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
    # Dictionary Vis
    ## Load Dictionary file
    path_Dict = os.path.join(path_exp, name_Dict)
    mat_Dict_DYAN = scipy.io.loadmat(path_Dict)
    Drr_Dict = mat_Dict_DYAN['Drr'] # 1x80
    num_poles_needed=Drr_Dict.shape[1]
    Dtheta_Dict = mat_Dict_DYAN['Dtheta'] # 1x80
    order_theta = np.argsort(np.insert(np.squeeze(Dtheta_Dict),0,0))
    ## Plot Dictionary with same color in the binary visualization
    Drr_temp = np.insert(np.squeeze(Drr_Dict),0,1)
    Dtheta_temp = np.insert(np.squeeze(Dtheta_Dict),0,0)
    ## First Pole
    axs_dict_color[0].set_title('FT Dict', fontsize=12)
    axs_dict_color[1].set_title('FT Dict Clean', fontsize=12)
    axs_dict_color[0].scatter( 0, 1, s=20, c=colorsys.hsv_to_rgb(0.0,1.0,1.0), alpha=0.8, edgecolors='none')
    axs_dict_color[1].scatter( 0, 1, s=20, c=colorsys.hsv_to_rgb(0.0,1.0,1.0), alpha=0.8, edgecolors='none')
    ## 
    list_id_NoRepeat=[0]
    list_poles_far=[(1,0)] # Must keep pole (1,0), 20211118
    for i_theta in order_theta:
        r_temp = Drr_temp[i_theta]
        theta_temp = Dtheta_temp[i_theta]
        
        S_temp = r_temp
        # H_temp = theta_temp/(2*np.pi) # instead of using 360 degree to normalize,use 180 degree to make color more different, 20211118
        H_temp = theta_temp/(4*np.pi/3)
        if H_temp < 0: H_temp=0
        elif H_temp > 1:  H_temp=1

        if S_temp < 0: S_temp=0
        elif S_temp > 1: S_temp=1
        color_temp=[colorsys.hsv_to_rgb(H_temp, S_temp, 1)]

        axs_dict_color[0].scatter( theta_temp, r_temp, s=20, c=color_temp, alpha=0.8, edgecolors='none')
        # axs_dict_color[0].scatter(-theta_temp, r_temp, s=20, c=color_temp, alpha=0.8, edgecolors='none')
        
        bool_close=False
        for r_far,theta_far in list_poles_far:
            if (distance_polar(r_temp,theta_temp,r_far,theta_far) < THD_distance) \
                or (distance_polar(r_temp,-theta_temp,r_far,theta_far) < THD_distance):
                bool_close=True
        if not bool_close:
            list_poles_far.append((r_temp,theta_temp))
            list_id_NoRepeat.append(i_theta)
            
            axs_dict_color[1].scatter( theta_temp, r_temp, s=20, c=color_temp, alpha=0.8, edgecolors='none')
            # axs_dict_color[1].scatter(-theta_temp, r_temp, s=20, c=color_temp, alpha=0.8, edgecolors='none')

    figs_dict_color.tight_layout()
    # figs_dict_color.savefig(os.path.join(dir_save,f"Dict_same-color-Bin_Ori-Clean.png"),dpi=400)

    # Plot Binarization with H-theta,S-rho,V-binarization/coefficients definite value
    dict_mat_bin={}
    for name_bin in list_name_C_B:
        view_exp = name_bin.split('_')[-2]
        path_bin = os.path.join(path_exp, name_bin)
        dict_mat_bin[view_exp] = scipy.io.loadmat(path_bin)
    # Loop over every action
    # # For binarization values
    # for id_act in range(len(mat_bin['bi_action'][0])):
    #     if id_act==0: continue
    #     bi_action_act=mat_bin['bi_action'][0,id_act]
    #     for id_vid in range(bi_action_act.shape[0]):
    #         bin_gumbel = np.squeeze(bi_action_act[id_vid])
    for name_joint, index_joint in dict_arm.items():
        for legend_dim_data, id_dim_data in dict_dim_data.items():
            if id_dim_data==1: continue
            # A1 - v2 | v3
            axs_dict_joint_cmp[0]=plot_subplot(dict_mat_bin['v2'], 0, 'v2', index_joint, id_dim_data, '^','r',
                                            Drr_temp,Dtheta_temp,num_poles_needed, num_used_top,
                                            axs_dict_joint_cmp[0])
            axs_dict_joint_cmp[0]=plot_subplot(dict_mat_bin['v3'], 0, 'v3', index_joint, id_dim_data, 'o','r',
                                            Drr_temp,Dtheta_temp,num_poles_needed, num_used_top,
                                            axs_dict_joint_cmp[0])
            legend_elements = [plt.scatter([], [], marker='^', color='r', label='View 2'),
                               plt.scatter([], [], marker='o', color='r', label='View 3')]
            axs_dict_joint_cmp[0].legend(handles=legend_elements,bbox_to_anchor =(1.1, -0.30), ncol = 2,fontsize=8) 
            axs_dict_joint_cmp[0].set_title(f'A1_V2-V3_{legend_dim_data}', fontsize=10)
            # A6 - v2 | v3
            axs_dict_joint_cmp[1]=plot_subplot(dict_mat_bin['v2'], 5, 'v2', index_joint, id_dim_data, '^','b',
                                            Drr_temp,Dtheta_temp,num_poles_needed, num_used_top,
                                            axs_dict_joint_cmp[1])
            axs_dict_joint_cmp[1]=plot_subplot(dict_mat_bin['v3'], 5, 'v3', index_joint, id_dim_data, 'o','b',
                                            Drr_temp,Dtheta_temp,num_poles_needed, num_used_top,
                                            axs_dict_joint_cmp[1])
            legend_elements = [plt.scatter([], [], marker='^', color='b', label='View 2'),
                               plt.scatter([], [], marker='o', color='b', label='View 3')]
            axs_dict_joint_cmp[1].legend(handles=legend_elements,bbox_to_anchor =(1.1, -0.30), ncol = 2,fontsize=8) 
            axs_dict_joint_cmp[1].set_title(f'A6_V2-V3_{legend_dim_data}', fontsize=10)
            # v3 - A1 | A6
            axs_dict_joint_cmp[2]=plot_subplot(dict_mat_bin['v3'], 0, 'v3', index_joint, id_dim_data, 'o','r',
                                            Drr_temp,Dtheta_temp,num_poles_needed, num_used_top,
                                            axs_dict_joint_cmp[2])
            axs_dict_joint_cmp[2]=plot_subplot(dict_mat_bin['v3'], 5, 'v3', index_joint, id_dim_data, 'o','b',
                                            Drr_temp,Dtheta_temp,num_poles_needed, num_used_top,
                                            axs_dict_joint_cmp[2])
            legend_elements = [plt.scatter([], [], marker='o', color='r', label='Action 1'),
                               plt.scatter([], [], marker='o', color='b', label='Action 6')]
            axs_dict_joint_cmp[2].legend(handles=legend_elements,bbox_to_anchor =(1.2, -0.30), ncol = 2,fontsize=8) 
            axs_dict_joint_cmp[2].set_title(f'V3_A1-A6_{legend_dim_data}', fontsize=10)
            figs_dict_joint_cmp.suptitle(f"{name_joint}",fontsize=12)
            figs_dict_joint_cmp.savefig(os.path.join(dir_save, f'{name_joint}.png'), dpi=400)
            for axs_temp in axs_dict_joint_cmp.flat: axs_temp.clear()

    pass


def main():
    try:
        path_exp=sys.argv[1]
    except:
        path_exp=dir_default

    # data = scipy.io.loadmat(path_mat_dic)
    # weightPoles(data['Coeff'], data['Drr'], data['Dtheta'], data['dictionary'],dim_data_draw=0)
    for i in range(1):
        weightPoles(path_exp, 
                    num_used_top=10, dim_data_draw=i,
                    theta_max_pi=True, weight_complex=False,
                    draw_weight=False, draw_traj=False, draw_cmap=False, draw_bin=True)


if __name__ == '__main__':
    main()
