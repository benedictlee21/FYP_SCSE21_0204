import torch
# from loss import *
from ChamferDistancePytorch.chamfer_python import distChamfer
import os
import h5py
import open3d as o3d
import numpy as np
# import pypcd
# import pypcd.pypcd.PointCloud
# import open3d as o3d
import time
import glob
"""
1. verify the distance between casecaded and pcn full, for test split

"""


root_dir = '/mnt/lustre/share/zhangjunzhe'

cascade_pathanme = os.path.join(root_dir,'cascade','test_data.h5')

data = h5py.File(cascade_pathanme, 'r')
gt = torch.from_numpy(data['complete_pcds'][()]).to('cuda')
partial = torch.from_numpy(data['incomplete_pcds'][()]).to('cuda')
labels = data['labels'][()]

pcn_test_pathname = '/mnt/lustre/share/zhangjunzhe/pcn/test.list'
pcn_test_dir = '/mnt/lustre/share/zhangjunzhe/pcn/test/complete'
pcn_test = open(pcn_test_pathname).readlines()

# offset2cat
offset_filepath = '/mnt/lustre/share/zhangjunzhe/topnet/synsetoffset2category.txt'
offset2cat  = {}
cat2offset = {}
cat_ordered_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']
offset2num = {}
with open(offset_filepath, 'r') as f:
        for line in f:
            ls = line.strip().split()
            offset2cat[ls[1]] = ls[0]
            cat2offset[ls[0]] = ls[1]
for i, cat in enumerate(cat_ordered_list):
    offset = cat2offset[cat]
    offset2num[offset] = i


### get the index of minimal distance
# index_list = []
# t1 = time.time()
# for j, stem in enumerate(pcn_test):
#     tic =  time.time()
#     stem = stem.rstrip('\n')
#     offset, _ = stem.split('/')
#     pathname = os.path.join(pcn_test_dir,stem+'.pcd')
#     pcd = o3d.io.read_point_cloud(pathname)
#     pcd_np  = np.asarray(pcd.points)
#     pcd_t = torch.from_numpy(pcd_np)
#     cat_num = offset2num[offset]
#     index_array = np.where(labels==cat_num)[0]
#     gt_cat = gt[index_array] # NOTE use this for partition the tensor
#     # gt_cat = gt[index_array]
#     cd_150 = []
#     for i in range(15):
#         pcd_tensor = pcd_t.cuda().unsqueeze(0).repeat(10,1,1)
#         dist1, dist2 , _, _ = distChamfer(gt_cat[10*i:10*(i+1)], pcd_tensor)
#         cd = dist1.mean(dim=1) + dist2.mean(dim=1)
#         cd_150.append(cd)
#     cd_150 = torch.cat(cd_150,0)
#     top_dist, idx = torch.topk(cd_150, 1, dim=0, largest=False)
#     # top_dist2, idx2 = torch.topk(cd_150, 2, dim=0, largest=False)
#     index_list.append(index_array[idx])
#     toc = time.time()
#     print('time spent',int(toc-tic))
#     if j > 0:
#         break
# print(index_list)
# t2 = time.time()


### assume algined, compute the distance, and obtained the largest
# cd_ls = []
# for j, stem in enumerate(pcn_test):
#     tic =  time.time()
#     stem = stem.rstrip('\n')
#     offset, _ = stem.split('/')
#     pathname = os.path.join(pcn_test_dir,stem+'.pcd')
#     pcd = o3d.io.read_point_cloud(pathname)
#     pcd_np  = np.asarray(pcd.points)
#     pcd_t = torch.from_numpy(pcd_np)
#     cat_num = offset2num[offset]
#     index_array = np.where(labels==cat_num)[0]
#     pcd_tensor = pcd_t.cuda().unsqueeze(0)
#     # import pdb; pdb.set_trace()
#     dist1, dist2 , _, _ = distChamfer(gt[j].unsqueeze(0), pcd_tensor)
#     cd = dist1.mean() + dist2.mean()
#     cd_ls.append(cd)
#     print(j)
# sorted(cd_ls)[-1] = 0.0003

# pathname = '/mnt/lustre/share/zhangjunzhe/pcn/test/complete/03001627/2842701d388dcd3d534fa06200d07790.pcd'

# import pdb; pdb.set_trace()

### compute partial and full
# cd1_ls = []
# for j, stem in enumerate(pcn_test):
#     tic =  time.time()
#     stem = stem.rstrip('\n')
#     offset, _ = stem.split('/')
#     pathname = os.path.join(pcn_test_dir,stem+'.pcd')
#     pcd = o3d.io.read_point_cloud(pathname)
#     pcd_np  = np.asarray(pcd.points)
#     pcd_t = torch.from_numpy(pcd_np)
#     cat_num = offset2num[offset]
#     index_array = np.where(labels==cat_num)[0]
#     pcd_tensor = pcd_t.cuda().unsqueeze(0)
#     # import pdb; pdb.set_trace()
#     dist1, dist2 , _, _ = distChamfer(partial[j].unsqueeze(0), pcd_tensor)
#     cd1 = dist1.mean() 
#     cd1_ls.append(cd1)
#     print(j)

# import pdb; pdb.set_trace()


### import 
input_dir = '/mnt/lustre/share/zhangjunzhe/topnet/test/partial/all'
pathnames = glob.glob(input_dir+"/*")
idx2pathname = {}
for pathname in pathnames:
    basename = os.path.basename(pathname)
    stem, txt = os.path.splitext(basename)
    idx2pathname[int(stem)] = pathname

dist_ls = []
dist2_ls = []
idx_ls = []
idx2_ls = []
for i in range(1200):
    tic = time.time()
    if i not in idx2pathname.keys():
        print(i,'not in topnet')
        continue
    
    pathname = idx2pathname[i]
    f = h5py.File(pathname, 'r')
    pcd = torch.from_numpy(np.array(f['data'])).cuda().unsqueeze(0)
    f.close()
    pcd = pcd.repeat(100,1,1)
    cd1_ls = []
    for j in range(12): 
        dist1, dist2 , _, _ = distChamfer(pcd, gt[j*100:(j+1)*100])
        cd1 = dist1.mean(dim=1) 
        cd1_ls.append(cd1)
    cd1_ls = torch.cat(cd1_ls,0)
    top_dist3, idx3 = torch.topk(cd1_ls, 2, dim=0, largest=False)
    # print(i)
    toc = time.time()
    # import pdb; pdb.set_trace()
    if top_dist3[0] * 5 < top_dist3[1]:
        print(i,'small than 5 times')
    else:
        print(top_dist3)

#
# 37 not in topnet
# 179 not in topnet
# 200 not in topnet
# 250 not in topnet
# 293 not in topnet
# 303 not in topnet
# 351 not in topnet
# 373 not in topnet
# 423 not in topnet
# 562 not in topnet
# 654 not in topnet
# 804 not in topnet
# 808 not in topnet
# 867 not in topnet
# 1006 not in topnet
# 1192 not in topnet