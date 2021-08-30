from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py

class CascadeShapeNetv1(data.Dataset):
    
    def __init__(self,split='train',class_choice='None',n_samples=0):

        self.DATA_PATH = '/mnt/lustre/share/zhangjunzhe/cascade' 
        self.split = split
        # self.catfile = './data/synsetoffset2category.txt'
        if self.split == 'train':
            basename = 'train_data.h5'
        elif self.split == 'test':
            basename = 'train_data.h5'
        elif self.split == 'valid':
            basename = 'train_data.h5'
        else:
            raise NotImplementedError
        # TODO should have mapping from offset / class --> digits
        # TODO assume we take the 1000 as of now
        pathname = os.path.join(self.DATA_PATH,basename)
        # gt_file_pathname = os.path.join(root_dir,'test_data.h5')
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]
        if class_choice.lower() == 'chair':
            # import pdb; pdb.set_trace()
            self.index_class_3 =  [i for (i, j) in enumerate(self.labels) if j == 3 ]
            if n_samples == 0:
                n_samples = len(self.index_class_3)
            # hard coded 1000
            np.random.seed(0)
            perm_list = np.random.permutation(len(self.index_class_3)).tolist()[:n_samples]
            self.index_list = perm_list
        elif class_choice == 'None':
            self.index = [i for i in range(len(list(self.labels)))]
            if n_samples == 0:
                n_samples = len(self.index)
            np.random.seed(0)
            perm_list = np.random.permutation(len(self.index)).tolist()[:n_samples]
            self.index_list = perm_list

        ### TODO change
        # self.gt = data['complete_pcds'][()][:1000]
        # self.partial = data['incomplete_pcds'][()][:1000]
        # self.labels = data['labels'][()][:1000]

    def __getitem__(self, index):
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx])
        # partial = torch.from_numpy(self.partial[full_idx])
        label = self.labels[index]
        
        return gt, label

    def __len__(self):
        return len(self.index_list)
        # return self.gt.shape[0]


class CascadeShapeNetv1_DDP(data.Dataset):
    
    def __init__(self,split='train',class_choice='None'):

        self.DATA_PATH = '/mnt/lustre/share/zhangjunzhe/cascade' 
        self.split = split
        # self.catfile = './data/synsetoffset2category.txt'
        if self.split == 'train':
            basename = 'train_data.h5'
        elif self.split == 'test':
            basename = 'test_data.h5'
        elif self.split == 'valid':
            basename = 'valid_data.h5'
        else:
            raise NotImplementedError
        # TODO should have mapping from offset / class --> digits
        # TODO assume we take the 1000 as of now
        pathname = os.path.join(self.DATA_PATH,basename)
        # gt_file_pathname = os.path.join(root_dir,'test_data.h5')
        data = h5py.File(pathname, 'r')
        self.gt = torch.from_numpy(data['complete_pcds'][()]).to('cuda')
        self.partial = torch.from_numpy(data['incomplete_pcds'][()]).to('cuda')
        self.labels = data['labels'][()]
        # import pdb; pdb.set_trace()
        self.index_class_3 =  [i for (i, j) in enumerate(self.labels) if j == 3 ]

        # hard coded 1000
        np.random.seed(0)
        perm_list = np.random.permutation(len(self.index_class_3)).tolist()[:1000]
        self.index_list = perm_list
        ### TODO change
        # self.gt = data['complete_pcds'][()][:1000]
        # self.partial = data['incomplete_pcds'][()][:1000]
        # self.labels = data['labels'][()][:1000]

    def __getitem__(self, index):
        full_idx = self.index_list[index]
        # gt = torch.from_numpy(self.gt[full_idx])
        # partial = torch.from_numpy(self.partial[full_idx])
        label = self.labels[index]
        gt = self.gt[index]
        
        return gt, label

    def __len__(self):
        return len(self.index_list)
        # return self.gt.shape[0]
