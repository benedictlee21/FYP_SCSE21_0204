from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py
import random

class CRNShapeNet(data.Dataset):
    """
    Dataset with GT and partial shapes provided by CRN
    Used for shape completion and pre-training tree-GAN
    """
    def __init__(self, args, classes_chosen = None):
        self.args = args
        self.classes_chosen = classes_chosen
        self.dataset_path = self.args.dataset_path
        self.class_choice = self.args.class_choice
        self.split = self.args.split
        
        print('CRN_dataset.py: __init__ - classes chosen:', classes_chosen)

        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')
        
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]
        
        np.random.seed(0)
        category_ordered_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']

        if classes_chosen is not None:
            category_id_list = []
            
            for each_class in classes_chosen:
                if each_class in category_ordered_list:
                    category_id_list.append(each_class)
            
            print('CRN_dataset multiclass category ID list:', category_id_list)
        else:
            category_id = category_ordered_list.index(self.class_choice.lower())
            #print('CRN_dataset single category ID:', cat_id)
        
        self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == category_id ])
        
        print('CRN_dataset.py - self.index_list:', self.index_list)
        print('Index list length:', len(self.index_list))
        
    def __getitem__(self, index):
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx])
        label = self.labels[index]
        partial = torch.from_numpy(self.partial[full_idx])
        return gt, partial, full_idx

    def __len__(self):
        return len(self.index_list)
