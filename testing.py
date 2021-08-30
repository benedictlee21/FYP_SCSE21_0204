import os
import sys
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # from collections import OrderedDict

import torch
# import torchvision.utils as vutils
# from data.dataset_benchmark import BenchmarkDataset
# # import utils
# from utils.dgp_utils import *
# from utils.treegan_utils import *
# from models import DGP
# from dgp import DGP
import os.path as osp
import time
# from loss import *
import h5py
import time
import numpy
from plots.completion_pcd_plot import draw_any_set




split = 'train'
DATA_PATH = '/mnt/lustre/share/zhangjunzhe/cascade' 
DATA_PATH = '/Users/zhangjunzhe/Desktop/pcd_datasets/cascaded'

# args.split = split
# catfile = './data/synsetoffset2category.txt'
if split == 'train':
    basename = 'train_data.h5'
elif split == 'test':
    basename = 'train_data.h5'
elif split == 'valid':
    basename = 'train_data.h5'
else:
    raise NotImplementedError
# TODO should have mapping from offset / class --> digits
# TODO assume we take the 1000 as of now
pathname = os.path.join(DATA_PATH,basename)
# gt_file_pathname = os.path.join(root_dir,'test_data.h5')
data = h5py.File(pathname, 'r')
gt = data['complete_pcds'][()]
partial = data['incomplete_pcds'][()]
labels = data['labels'][()]
# import pdb; pdb.set_trace()
# index_class_0 =  [i for (i, j) in enumerate(labels) if j == 0 ]
dict_of_list = {} # key is 0 - 7, value is long list
for cls_idx in range(8):
    dict_of_list[cls_idx] = [i for (i, j) in enumerate(labels) if j == cls_idx ]

# print (length)
for key, value in dict_of_list.items():
    print(key, 'cnt', len(value))

# 0 cnt 3795
# 1 cnt 1322
# 2 cnt 5677
# 3 cnt 5750
# 4 cnt 2068
# 5 cnt 2923
# 6 cnt 5750
# 7 cnt 1689
print('sleeping')
time.sleep(60)

for key, value in dict_of_list.items():
    print('doing key', key)
    for i in value[:20]:
        flag_list =['gt', 'partial']
        pcd_list_numpy = [gt[i],partial[i]]
        pcd_list = [torch.from_numpy(pcd) for pcd in pcd_list_numpy]
        # output_dir = '/Users/zhangjunzhe/Downloads/pointsets/benchmark/drawings_t'
        output_dir = '/Users/zhangjunzhe/Downloads/pointsets/benchmark/'+str(key)
        output_stem = str(i)
        draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(1,2))
        

print('sleeping')
time.sleep(60)
#####

root_dir = '/Users/zhangjunzhe/Downloads/pointsets/benchmark'

basenames =  ['best_cd_t_network_pcds.h5', 'best_cd_p_network_pcds.h5']
models = ['cascade', 'pcn', 'msn', 'topnet']

gt_file_pathname = os.path.join(root_dir,'test_data.h5')

other_file_pathnames_t = [os.path.join(root_dir, model, basenames[0]) for model in models]
other_file_pathnames_p = [os.path.join(root_dir, model, basenames[1]) for model in models]
print(gt_file_pathname)
print(other_file_pathnames_t)
print(other_file_pathnames_p)

input_file = h5py.File(gt_file_pathname, 'r')
gt_data = input_file['complete_pcds'][()]
input_data = input_file['incomplete_pcds'][()]

output_files = []
output_data = []
# for pathname in other_file_pathnames_t:
for pathname in other_file_pathnames_p:
    # output_files.append(h5py.File(pathname, 'r'))
    output_file = h5py.File(pathname, 'r')
    output_data.append(output_file['output_pcds'][()])

n = 1200
for i in range(n):
    flag_list =['gt', 'partial'] + models
    pcd_list_numpy = [gt_data[i], input_data[i]] + [ output[i] for output in output_data]
    pcd_list = [torch.from_numpy(pcd) for pcd in pcd_list_numpy]
    # output_dir = '/Users/zhangjunzhe/Downloads/pointsets/benchmark/drawings_t'
    output_dir = '/Users/zhangjunzhe/Downloads/pointsets/benchmark/drawings_p'
    output_stem = str(i)
    draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(2,3))
    if i > 150:
        break
