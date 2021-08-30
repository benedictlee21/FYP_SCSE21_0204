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
import json
"""
check the point range 
"""
cascade = False
### for cascade 
if cascade:
    root_dir = '/mnt/lustre/share/zhangjunzhe'

    cascade_pathanme = os.path.join(root_dir,'cascade','train_data.h5')

    data = h5py.File(cascade_pathanme, 'r')
    gt = torch.from_numpy(data['complete_pcds'][()])
    partial = torch.from_numpy(data['incomplete_pcds'][()])
    labels = data['labels'][()]


    max_loc, _ = gt.max(dim=1)
    min_loc, _ = gt.min(dim=1)

    print('max of max\n',max_loc.max(0))
    print('min of max\n',max_loc.min(0))
    print('max of min\n',min_loc.max(0))
    print('min of min\n',min_loc.min(0))
    # max of max values=tensor([0.4951, 0.4986, 0.4878]),
    # min of max values=tensor([0.0284, 0.0263, 0.0142]),
    # max of min values=tensor([-0.4954, -0.4987, -0.4878]),
    # values=tensor([-0.0283, -0.0268, -0.0142]),



### check v0 segementation
shapenetcore = True
tic = time.time()
if shapenetcore:
    root_dir = '/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0'
    pathnames = []
    json_file = 'train_test_split/shuffled_train_file_list.json'


    json_pathname = os.path.join(root_dir,json_file)
    with open(json_pathname, 'r') as file:
        raw_pathnames = json.load(file)
    for raw_name in raw_pathnames:
        a,b,c = raw_name.split('/')
        if b != '03001627':
            continue
        pathname = os.path.join(root_dir,b,'points',c+'.pts')
        # print(pathname)
        pathnames.append(pathname)
        # import pdb; pdb.set_trace() 
# make the numpy as tensor
toc = time.time()
print('time for pathnames',toc-tic)
# pcd_ls = []
max_ls = []
min_ls = []
for i, pathname in enumerate(pathnames):
    # if i%100 > 0:
        # continue
    pcd = np.loadtxt(pathname).astype(np.float32)
    pcd = torch.from_numpy(pcd)
    max_loc, _ = pcd.max(dim=0)
    min_loc, _ = pcd.min(dim=0)
    # seg = np.loadtxt(fn[2]).astype(np.int64)
    # pcd_ls.append(pcd)
    max_ls.append(max_loc)
    min_ls.append(min_loc)
    

max_ls = torch.stack(max_ls)
min_ls = torch.stack(min_ls)
print('time for load pcs 1/100', int(time.time()-toc))
# max_loc, _ = gt.max(dim=1)
# min_loc, _ = gt.min(dim=1)

print('max of max\n',max_ls.max(0))
print('min of max\n',max_ls.min(0))
print('max of min\n',min_ls.max(0))
print('min of min\n',min_ls.min(0))
import pdb; pdb.set_trace() 
# max of max
#  torch.return_types.max(
# values=tensor([0.4920, 0.4884, 0.4586]),
# indices=tensor([100,   5,  67]))
# min of max
#  torch.return_types.min(
# values=tensor([0.0163, 0.0490, 0.0102]),
# indices=tensor([ 59, 100,   0]))
# max of min
#  torch.return_types.max(
# values=tensor([-0.0214, -0.0492, -0.0102]),
# indices=tensor([ 32, 100,   0]))
# min of min
#  torch.return_types.min(
# values=tensor([-0.4951, -0.4900, -0.4586]),
# indices=tensor([100,   5,  67]))

# only for chair
# max of max
#  torch.return_types.max(
# values=tensor([0.4490, 0.4711, 0.4807]),
# indices=tensor([1192,  921, 2067]))
# min of max
#  torch.return_types.min(
# values=tensor([0.1004, 0.0939, 0.0837]),
# indices=tensor([2067, 2067, 2519]))
# max of min
#  torch.return_types.max(
# values=tensor([-0.1004, -0.0917, -0.0837]),
# indices=tensor([2067, 2067, 2519]))
# min of min
#  torch.return_types.min(
# values=tensor([-0.4489, -0.4712, -0.4807]),
# indices=tensor([1192,  921, 2067]))