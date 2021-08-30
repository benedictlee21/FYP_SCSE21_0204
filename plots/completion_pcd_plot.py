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

def draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=None):
    """
    this should be more flexible to draw point clouds 
    directly called from training 
    horizontally
    squeeze if pcd is in shape [1, 2048, 3]
    """

    ax_min = 0
    ax_max = 0
    pcd_np_list = []
    for pcd in pcd_list:
        if pcd.shape[0] == 1:
            pcd.squeeze_(0)
        pcd = pcd.detach().cpu().numpy()
        pcd_np_list.append(pcd)
        ax_min = min(ax_min, np.min(pcd))
        ax_max = max(ax_max, np.max(pcd))
    ax_limit = max(abs(ax_min),ax_max) * 1.05  

    if layout == None:
        row = 1
        col = len(pcd_np_list)
        fig = plt.figure(figsize=(len(pcd_list)*4, 4))
    else:
        row, col = layout
        fig = plt.figure(figsize=(col*4, row*4))

    for i in range(len(pcd_np_list)):
        pcd = pcd_np_list[i]
        ax = fig.add_subplot(row,col,i+1,projection='3d')
        ax.scatter(pcd[:,0],pcd[:,2],pcd[:,1],s=1,label=flag_list[i])
        ax.set_xlim([-ax_limit,ax_limit])
        ax.set_ylim([-ax_limit,ax_limit])
        ax.set_zlim([-ax_limit,ax_limit ])
        ax.set_title(flag_list[i])

    output_f = os.path.join(output_dir,output_stem+'.png')
    plt.savefig(output_f)

def draw_one_set(input_dir,output_dir, id, draw_mode):
    # refer link : https://matplotlib.org/3.1.1/gallery/mplot3d/subplot3d.html
    # pcd_pathname1 = '/Users/stsgdzsb19060007/Downloads/pointsets/knn/knn6/20.txt'
    # pcd_pathname2 = '/Users/stsgdzsb19060007/Downloads/pointsets/knn/knn6/25.txt'
    # pcd_pathname3 = '/Users/stsgdzsb19060007/Downloads/pointsets/knn/knn6/30.txt'
    # pcd_pathname4 = '/Users/stsgdzsb19060007/Downloads/pointsets/knn/knn6/35.txt'
    pcd_pathname1 = os.path.join(input_dir,id+'_target.txt')
    pcd_pathname2 = os.path.join(input_dir,id+'_xmap.txt')
    pcd_pathname3 = os.path.join(input_dir,id+'_x.txt')
    pcd_pathname4 = os.path.join(input_dir,id+'_gt.txt')
    
    try: 
        pcd1 = np.loadtxt(pcd_pathname1,delimiter=';').astype(np.float32)
        pcd2 = np.loadtxt(pcd_pathname2,delimiter=';').astype(np.float32)
        pcd3 = np.loadtxt(pcd_pathname3,delimiter=';').astype(np.float32)
        pcd4 = np.loadtxt(pcd_pathname4,delimiter=';').astype(np.float32)
    except:
        print('pathname1',pcd_pathname1)
        print('pathname2',pcd_pathname2)
        print('pathname3',pcd_pathname3)
        print('pathname4',pcd_pathname4)
    # xs = pcd[:,0]
    # ys = pcd[:,2]
    # zs = pcd[:,1]
    # import pdb; pdb.set_trace()
    ax_min = np.min(pcd4)
    ax_max = np.max(pcd4)
    ax_limit = max(abs(ax_min),ax_max) * 1.05

    if draw_mode == '2x2':
        target_title = 'target' 
        row = 2
        col = 2 
        fig = plt.figure(figsize=(8, 8))
    elif draw_mode == '1x3':
        target_title = 'input'
        row = 1
        col = 3
        fig = plt.figure(figsize=(12, 4))
    

    
    
    i = 1
    ax = fig.add_subplot(row,col,i,projection='3d')
    ax.scatter(pcd1[:,0],pcd1[:,2],pcd1[:,1],s=1,label=target_title)
    ax.set_xlim([-ax_limit,ax_limit])
    ax.set_ylim([-ax_limit,ax_limit])
    ax.set_zlim([-ax_limit,ax_limit ])
    ax.set_title(target_title)

    if draw_mode == '2x2':
        i+=1
        ax = fig.add_subplot(row,col,i,projection='3d')
        ax.scatter(pcd2[:,0],pcd2[:,2],pcd2[:,1],s=1,label='x_map')
        ax.set_xlim([-ax_limit,ax_limit])
        ax.set_ylim([-ax_limit,ax_limit])
        ax.set_zlim([-ax_limit,ax_limit ])
        ax.set_title('x_map')
    
    
    i+=1
    ax = fig.add_subplot(row,col,i,projection='3d')
    ax.scatter(pcd3[:,0],pcd3[:,2],pcd3[:,1],s=1,label='output')
    ax.set_xlim([-ax_limit,ax_limit])
    ax.set_ylim([-ax_limit,ax_limit])
    ax.set_zlim([-ax_limit,ax_limit ])
    ax.set_title('output')

    i+=1
    ax = fig.add_subplot(row,col,i,projection='3d')
    ax.scatter(pcd4[:,0],pcd4[:,2],pcd4[:,1],s=1, label='GT')
    ax.set_xlim([-ax_limit,ax_limit])
    ax.set_ylim([-ax_limit,ax_limit])
    ax.set_zlim([-ax_limit,ax_limit ])
    ax.set_title('GT')

    # plt.show()
    output_f = os.path.join(output_dir,id+'.png')
    plt.savefig(output_f)


def draw_all_sets(input_dir, output_dir, draw_mode):
    pathnames = glob.glob(input_dir+"/*")
    ids = set()
    
    for pathname in pathnames:
        stem, ext = os.path.splitext(os.path.basename(pathname))
        stem_id = stem.split('_')[0]
        ids.add(stem_id)
        # print(stem_id)
    # import pdb; pdb.set_trace()
    # ids = set(ids)
    print(ids)
    for stem_id in ids:
        draw_one_set(input_dir,output_dir, stem_id, draw_mode)



if __name__ == '__main__':
    draw_mode = '2x2'
    # draw_mode = '1x3'

    input_dir  = '/mnt/lustre/zhangjunzhe/pcd_lib/saved_samples/invert_topnet_scan2'
    # input_dir = '/Users/stsgdzsb19060007/Downloads/pointsets/baselines/completion/invert_knn_hole_5x200'
    # input_dir_base = '/mnt/lustre/zhangjunzhe/pcd_lib/saved_samples/'
    if draw_mode == '2x2':
        output_dir = input_dir+'_visual'
    elif draw_mode == '1x3':
        output_dir = input_dir+'_visual_1x3'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    tic = time.time()
    draw_all_sets(input_dir,output_dir,draw_mode)
    toc = time.time()
    print('drawing done in',int(toc-tic),'s')
