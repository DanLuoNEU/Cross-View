import numpy as np
import os
import math

dataType = '2D'
root_skeleton="/data/Yuexi/NTU-RGBD/skeletons/npy"
# root_skeleton = '/data/NTU-RGBD/poses'
name_samples = os.listdir(root_skeleton)
delList = []
i = 0
for name_sample in name_samples:
    skeleton = np.load(os.path.join(root_skeleton, name_sample), allow_pickle=True).item()[
                'rgb_body0']
    T_sample,_,_ = skeleton.shape
    print('preprocess sample:', i)
    for t in range(0, T_sample):

        for data_joint in skeleton[t]:
        # data_joint = skeleton[t]
        #     if math.isnan(data_joint[0]) or math.isnan(data_joint[1]):
            if np.isnan(data_joint[0]) == True or np.isnan(data_joint[1]) == True:
                # print(name_sample)
                print('sample:',name_sample, 'joint:', data_joint)
                delList.append(name_sample)
                # break
            if data_joint[0] == 0 and data_joint[1]== 0:
                delList.append(name_sample)

            # print('check')
            # else:
            #     # continue
            #     break
    i += 1
delList = np.unique(np.array(delList))
# np.savez('./NTU_deadList_120', x=delList)

print('done')