import matplotlib.pyplot as plt
import os
import numpy as np

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

    return np.array(sparse_data)

def plot_stats(data_abs, data_sqr, n_bins=10):

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    print('data shape ',data_sqr.shape)
    axs[0].set_xlim([0, 1])
    axs[0].set_xlabel("range of abs C values")
    axs[0].set_title("Frequency of abs C values")
    axs[0].set_ylabel("Frequency")
    axs[0].hist(data_abs, bins=n_bins)    

    axs[1].set_xlim([0, 1])
    axs[1].set_xlabel("range of C squared values")
    axs[1].set_title("Frequency of C squared values")
    axs[1].set_ylabel("Frequency")
    axs[1].hist(data_sqr, bins=n_bins)    

    plt.show()
    #plt.savefig(save_path)


if __name__ == "__main__":

    lst_path = 'sparse_data_c2.lst'
    lst_reader = open(lst_path, 'r')

    sparse_data = read_data(lst_reader, delimiter=',')
    sparse_data = sparse_data.reshape((sparse_data.shape[0], 161, -1))
    sparse_data_abs = abs(sparse_data)
    sparse_data_sqr = sparse_data.copy()**2

    for num in range(sparse_data.shape[0]):
        print('min max ', np.min(sparse_data[num,:]), np.max(sparse_data[num,:]))
        plot_stats(sparse_data_abs[num,:], sparse_data_sqr[num,:], n_bins=10)
