# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

### ref: https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np
import os
import glob
import time
import h5py

def draw_one_pair_from_h5_overlap(gt_pathname, partial_pathname):
    with h5py.File(partial_pathname, 'r') as f:
        pcd_partial = np.array(f['data'])
    with h5py.File(gt_pathname, 'r') as f:
        pcd_gt = np.array(f['data'])
    
    ax_min = np.min(pcd_gt)
    ax_max = np.max(pcd_gt)
    ax_limit = max(abs(ax_min),ax_max) * 1.05

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pcd_gt[:,0],pcd_gt[:,2],pcd_gt[:,1],s=1, label='GT')

    ax.scatter(pcd_partial[:,0],pcd_partial[:,2],pcd_partial[:,1],s=1, color ='r', label='partial')
    ax.set_xlim([-ax_limit,ax_limit])
    ax.set_ylim([-ax_limit,ax_limit])
    ax.set_zlim([-ax_limit,ax_limit ])

    plt.show()
# if mode == 1:
    # colors = cm.rainbow(np.linspace(0,1,n_groups))
    # for i,c in enumerate(colors):
        # if i > 5:
        #     continue
        # xs = pcd[i*n_per_group:(i+1)*n_per_group,0]
        # ys = pcd[i*n_per_group:(i+1)*n_per_group,2]
        # zs = pcd[i*n_per_group:(i+1)*n_per_group,1]
        # ax.scatter(xs, ys, zs,s=5,color=c,label=str(i))
        # cnt+= xs.shape[0]


def draw_one_pair_from_h5(gt_pathname, partial_pathname):
    # refer link : https://matplotlib.org/3.1.1/gallery/mplot3d/subplot3d.html
    with h5py.File(partial_pathname, 'r') as f:
        pcd_partial = np.array(f['data'])
    with h5py.File(gt_pathname, 'r') as f:
        pcd_gt = np.array(f['data'])

    ax_min = np.min(pcd_gt)
    ax_max = np.max(pcd_gt)
    ax_limit = max(abs(ax_min),ax_max) * 1.05

    row = 1
    col = 2

    fig = plt.figure(figsize=(12, 6))
    
    i = 1
    ax = fig.add_subplot(row,col,i,projection='3d')
    ax.scatter(pcd_gt[:,0],pcd_gt[:,2],pcd_gt[:,1],s=1,label='GT')
    ax.set_xlim([-ax_limit,ax_limit])
    ax.set_ylim([-ax_limit,ax_limit])
    ax.set_zlim([-ax_limit,ax_limit ])
    ax.set_title('GT')

   
    i=2
    ax = fig.add_subplot(row,col,i,projection='3d')
    ax.scatter(pcd_partial[:,0],pcd_partial[:,2],pcd_partial[:,1],s=1,label='partial')
    ax.set_xlim([-ax_limit,ax_limit])
    ax.set_ylim([-ax_limit,ax_limit])
    ax.set_zlim([-ax_limit,ax_limit ])
    ax.set_title('partial')

    plt.show()



### read basenames of the v0 - in my GI
file1 = open("basenames.txt","r")
v0_basenames = []
for line in file1:
    # print(line)
    # basename = os.path.basename(line.rstrip('\n'))
    # print(basename)
    stem, ext = line.rstrip('\n').split('.')
    # print(stem)
    v0_basenames.append(stem+'.h5')
    # print(line[:4]+'.h5')

# i = 1
# for basename in v0_basenames:
#     print(i,'  ',basename)
#     i+=1


# input_dir = '~/Downloads/temp/pcd_datasets/topnet/train'
input_dir = '/Users/zhangjunzhe/Desktop/pcd_datasets/topnet/train'
cls_offset = '/03001627'
# print(input_dir+'/partial'+cls_offset)
# print(os.path.isdir(input_dir+'/partial'+cls_offset))
pathnames = glob.glob(input_dir+'/partial'+cls_offset+"/*")

basenames = [os.path.basename(pathname) for pathname in pathnames]
# print(basenames[0])
print(len(pathnames))

i = 1000
basename = basenames[i]

# basename = v0_basenames[68]  
basename = 'b455c3037b013d85492d9da2668ec34c.h5'
# good ones: 12, 10, 17, 85, 84, 86
# no found: 30, 4

pathname_patial = input_dir + '/partial' + cls_offset + '/' +  basename
pathname_gt = input_dir + '/gt' + cls_offset + '/' + basename

print(basename)
draw_one_pair_from_h5(pathname_gt, pathname_patial)
# draw_one_pair_from_h5_overlap(pathname_gt, pathname_patial)

# 2 3f23ed37910bbb589eb12cef6be5d578.h5
# 10 da292ce4ccdcfb2c842c47c8032438a1 (very partial, can only overfit)
# 100 6c25ec1178e9bab6e545858398955dd1 (doable)
# 500 8e21f7a88cecf4ab5ef2246d39b30aec.h5 (not recognizable)
