import matplotlib.pyplot as plt
import os
import numpy as np
np.set_printoptions(suppress=True)

def write_data(lst_writer, data, delimiter=','):

    data = data.reshape(data.shape[0]*data.shape[1]*data.shape[2])
    data_lst = data.tolist()

    lst_writer.write(delimiter.join(str(j) for j in data_lst))
    lst_writer.write('\n')

def read_data(lst_reader, delimiter=','):

    sparse_data = []
    lines = lst_reader.readlines()
    for line in lines:
        words = line.strip('\n').split(delimiter)
        anns = [float(x) for x in words]
        sparse_data.append(anns)

    sparse_data.reverse()

    return np.array(sparse_data)

def plot_stats(data_abs, data_sqr, num_joint, n_bins=10):

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    print('data shape ',data_sqr.shape)
    axs[0].set_xlim([0, 1])
    axs[0].set_xlabel("range of abs C values")
    axs[0].set_title("Hist of abs(C) values for joint " + str(num_joint))
    axs[0].set_ylabel("Frequency")
    axs[0].hist(data_abs, bins=n_bins)    

    axs[1].set_xlim([0, 1])
    axs[1].set_xlabel("range of C squared values")
    axs[1].set_title("Hist of C**2 values for joint " + str(num_joint))
    axs[1].set_ylabel("Frequency")
    axs[1].hist(data_sqr, bins=n_bins)    

    #plt.savefig(plot_dir + '/' + str(num_joint) + '.png')

def calc_conf_mat(sparse_data, binary_data, thresh=0.02):

    label = (sparse_data > thresh).astype('int')
    zeros = np.zeros(sparse_data.shape)

    c0_ids = np.where(label == 0)[0]
    c1_ids = np.where(label == 1)[0]

    tp = np.multiply(label == 1, binary_data == 1).astype('int')
    tn = np.multiply(label == 0, binary_data == 0).astype('int')
    fp = np.multiply(label == 0, binary_data == 1).astype('int')
    fn = np.multiply(label == 1, binary_data == 0).astype('int')

    tp_ids = np.where(tp == 1)[0]
    tn_ids = np.where(tn == 1)[0]
    fp_ids = np.where(fp == 1)[0]
    fn_ids = np.where(fn == 1)[0]

    print('true negatives: ',len(tn_ids)/len(c0_ids))
    print('true positives: ',len(tp_ids)/len(c1_ids))
    print('false negatives: ',len(fn_ids)/len(c1_ids))
    print('false positives: ',len(fp_ids)/len(c0_ids))

    fig1, axs1 = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # label
    axs1[0].scatter(sparse_data[c0_ids], zeros[c0_ids], s=5, color = 'red', label = 'class 0',marker='*')
    axs1[0].scatter(sparse_data[c1_ids], zeros[c1_ids], s=5, color = 'blue', label = 'class 1', marker='*')
    axs1[0].set_title('True labels')
    axs1[0].set_xlabel('C**2 value distribution')
    axs1[0].legend(["class 0 ", 'class 1'])
    axs1[0].set_ylim([-0.01, +0.01])

    # prediction    
    axs1[1].plot(sparse_data[tn_ids], zeros[tn_ids]-0.0025, '+', color ='g', markersize = 6)
    axs1[1].plot(sparse_data[tp_ids], zeros[tp_ids]-0.0025, '.', color ='g', markersize = 6)
    axs1[1].plot(sparse_data[fp_ids], zeros[fp_ids]+0.0025, '+', color ='r', markersize = 6)
    axs1[1].plot(sparse_data[fn_ids], zeros[fn_ids]+0.0025, '.', color ='r', markersize = 6)
    axs1[1].legend(["class 0 correctly classified", 'class 1 correctly classified','class 0 wrongly classified', 'class 1 wrongly classified'])
    axs1[1].set_ylim([-0.01, +0.01])
    axs1[1].set_title('Classification results')
    axs1[1].set_xlabel('C**2 value distribution')
    

if __name__ == "__main__":

    lst_path = 'old_bin/sparse_data_c2.lst'
    lst_reader = open(lst_path, 'r')

    bin_lst_path = 'old_bin/sparse_data_b2.lst'
    bin_lst_reader = open(bin_lst_path, 'r')

    combine_joints = 1
    plot_dir = '/home/ubuntu/Documents/US/NEU/RA/CS_CV/joints/'

    sparse_data = read_data(lst_reader, delimiter=',')
    sparse_data = sparse_data.reshape((sparse_data.shape[0], 161, -1))
    num_joints = sparse_data.shape[2]
    print('sparse_data shape: ',sparse_data.shape)

    binary_data = read_data(bin_lst_reader, delimiter=',')
    binary_data = binary_data.reshape((binary_data.shape[0], 161, -1))
    num_joints = binary_data.shape[2]
    print('binary_data shape: ',binary_data.shape)
    
    sparse_data_abs = abs(sparse_data)
    sparse_data_sqr = sparse_data.copy()**2

    for num in range(sparse_data.shape[0]):

        if combine_joints:
            for num_joint in range(num_joints):

                print('joint_num min max ', num_joint, np.min(sparse_data[num,:,num_joint]), np.max(sparse_data[num,:,num_joint]))
                print('sparse data ',np.round(sparse_data[num,:,num_joint], 4))
                print('binary data ',binary_data[num,:,num_joint])
                plot_stats(sparse_data_abs[num,:,num_joint], sparse_data_sqr[num,:,num_joint], num_joint)
                calc_conf_mat(sparse_data_sqr[num,:,num_joint], binary_data[num,:,num_joint])
                plt.show()
                
        else:
            print('min max ', np.min(sparse_data[num,:]), np.max(sparse_data[num,:]))
            plot_stats(sparse_data_abs[num,:], sparse_data_sqr[num,:], n_bins=10)
