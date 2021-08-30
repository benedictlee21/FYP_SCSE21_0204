# adapt from ref: https://github.com/lynetcha/completion3d/blob/1dc8ffac02c4ec49afb33c41f13dd5f90abdf5b7/shared/datasets/shapenet.py#L15

from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py

def shapenetv1_list2dict(list_pathname,offset2cat_pathname):
    """
    given a data split in the form of .list, say train.list
    to convert it into a dict of sorted lists, class name being the key
    which can be used for sampling by a seed (considering partially trained)
    NOTE: designed for v1, but seemed also can be used in v0
    """
    # obtain list of offset/basename, eg, 
    # '04530566/7d7fe630419cb3e3ff217e1345ac0f8\n'
    pcs_ls = open(list_pathname).readlines()
    
    # get offset2cat dict
    offset2cat = {}
    cat2offset = {}
    with open(offset2cat_pathname, 'r') as f:
        for line in f:
            ls = line.strip().split()
            offset2cat[ls[1]] = ls[0]
            cat2offset[ls[0]] = ls[1]
    v1_8_cats = ['plane', 'cabinet', 'car', 'chair', 'lamp','couch', 'table', 'watercraft']
    
    # get empty dict of lists
    cat2list = {}
    # for offset, cat in offset2cat.items(): # this has 13 classes
    for cat in v1_8_cats:
        cat2list[cat] = []
    
    # append the basename stems into corresponding cat
    for itm in pcs_ls:
        offset, stem = itm.split('/')
        stem = stem.rstrip('\n')
        cat2list[offset2cat[offset]].append(stem)
    
    # at the end, sort the lists in the dict
    for cat, unsorted_list in cat2list.items():
        cat2list[cat] = sorted(unsorted_list)
    # import pdb; pdb.set_trace()
    return cat2list,offset2cat, cat2offset



class TopNetShapeNetv1(data.Dataset):
    def __init__(self,split='train',class_choice='None'):

        self.DATA_PATH = '/mnt/lustre/share/zhangjunzhe/topnet/shapenet' 
        self.split = split
        self.catfile = './data/synsetoffset2category.txt'





# data = TopNe
data_path = '/mnt/lustre/share/zhangjunzhe/topnet/shapenet'
# list_pathname = data_path
# split = 'train'
# lists = open(data_path + '/%s.list' % (split)).readlines()
# import pdb; pdb.set_trace()

list_pathname = data_path + '/train.list'
 
offset2cat_pathname = data_path + '/synsetoffset2category.txt'

cat2list, offset2cat, cat2offset = shapenetv1_list2dict(list_pathname,offset2cat_pathname)
print('done flag')

new_dir = '/mnt/lustre/share/zhangjunzhe/topnet/shapenet/train/partial'

offset = '03001627'
pcd_cls = 'chair'
basename = cat2list['chair'][0]
pathname = os.path.join(new_dir,offset,basename+'.h5')
# print('isdir',os.path.isdir(pathname))
print(pathname)
f = h5py.File(pathname, 'r')
pcd = np.array(f['data'])
import pdb; pdb.set_trace()
f.close()

print(pcd.shape)


