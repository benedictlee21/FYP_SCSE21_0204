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
    def __init__(self, args, classes_chosen = None, is_eval = False):
        self.args = args
        self.classes_chosen = classes_chosen
        self.dataset_path = self.args.dataset_path
        self.class_choice = self.args.class_choice
        
        print('CRN_dataset.py: __init__ - classes chosen:', classes_chosen)
        
        # Value of string 'split' determines which '.h5' dataset file is used, either training or test.
        self.split = self.args.split
        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')
        
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]
        
        category_ordered_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']
        print('Category ordered list:', category_ordered_list)

        # Import shapes from input dataset for multiclass.
        if classes_chosen is not None:
        
            # Perform remapping of input class list to match the dataset class index:
            consolidated_classes = np.array([0] * len(category_ordered_list))
            remapped_classes = np.array([0] * len(category_ordered_list))
            
            # Compile all the indexes with '1' in them into a single array for multiclass training.
            for sub_array in classes_chosen:
                for index in range(len(sub_array)):
                    if sub_array[index] == 1:
                        consolidated_classes[index] = 1
            print('Consolidated array:', consolidated_classes)
            
            # Mapping: Input index > Output index
            # Chair: 0 > 3
            # Table: 1 > 6
            # Couch: 2 > 5
            # Cabinet: 3 > 1
            # Lamp: 4 > 4
            # Car: 5 > 2
            # Plane: 6 > 0
            # Watercraft: 7 > 7
            
            # Perform remapping of all selected class indexes to the training dataset indexes:
            remapped_classes[3] = consolidated_classes[0]
            remapped_classes[6] = consolidated_classes[1]
            remapped_classes[5] = consolidated_classes[2]
            remapped_classes[1] = consolidated_classes[3]
            remapped_classes[4] = consolidated_classes[4]
            remapped_classes[2] = consolidated_classes[5]
            remapped_classes[0] = consolidated_classes[6]
            remapped_classes[7] = consolidated_classes[7]
            
            print('Input: chair, table, couch, cabinet, lamp, car, plane, watercraft ->', classes_chosen)
            print('REMAPPED TO:')
            print('Output:', category_ordered_list, '->', remapped_classes)
            self.category_indexes = []
            
            # For each class category chosen, append its index to the list.
            for index in range(len(remapped_classes)):
                if remapped_classes[index] == 1:
                    self.category_indexes.append(index)
            
            print('CRN_dataset multiclass category ID list by index:', self.category_indexes)
            
            # Master list to append all indexes from each class into.
            self.index_list = []
            
            for index in self.category_indexes:
                self.one_class_index = np.array([shape for (shape, class_index) in enumerate(self.labels) if class_index == index])
                
                # View the indexes and shapes for each individual class used in multiclass.
                print('Category index:', index)
                print('Shape indexes:', self.one_class_index)
                
                # May need to reduce the training dataset size for pretraining.
                # Use the full test dataset size for evaluation.
                if not is_eval:
                
                    # Convert the numpy array into a list before sampling.
                    self.one_class_index = self.one_class_index.tolist()
                
                    print('Pretraining - randomly sampling', self.args.samples_per_class, 'out of 5750 shapes per class')
                    
                    # Get input argument for the number of samples per class to use for multiclass pretraining.
                    self.one_class_index = random.sample(self.one_class_index, self.args.samples_per_class)
                
                for index in self.one_class_index:
                    self.index_list.append(index)
            
            # Shuffle the class indexes in the master list at random to ensure even model pretraining across all classes.
            # Set the random seed to obtain reproducible results.
            #np.random.seed(0)
            
            if self.args.split == 'train':
                print('Shuffling dataset indexes.')
                random.shuffle(self.index_list)
                
            #print('CRN multiclass shuffled index list:', self.index_list)
            
            # Convert the list back into a numpy array.
            self.index_list = np.array(self.index_list)
            print('Multiclass index list length:', len(self.index_list))
            
        # Import shapes from input dataset for single class.
        elif self.class_choice != 'multiclass':
            # Obtain a single category ID corresponding to the single class.
            category_id = category_ordered_list.index(self.class_choice.lower())
            #print('CRN_dataset single category ID:', cat_id)
        
            # Extract all the shapes from the training dataset that match the single class index.
            self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == category_id ])
            
            # May need to reduce the training dataset size for pretraining.
            # Use the full test dataset size for evaluation.
            if not is_eval:
                # Convert the numpy array into a list before sampling.
                self.index_list = self.index_list.tolist()
        
                # Use fewer samples for training if time or computational resources are limited.
                self.index_list = random.sample(self.index_list, self.args.samples_per_class)
            
                # Convert the list back into a numpy array.
                self.index_list = np.array(self.index_list)
        
            #print('CRN Single class index list:', self.index_list)
            print('Single class index list length:', len(self.index_list))
        
    # Uses the index list created during the class initialization.
    def __getitem__(self, index):
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx])
        label = self.labels[index]
        partial = torch.from_numpy(self.partial[full_idx])
        return gt, partial, full_idx

    def __len__(self):
        return len(self.index_list)
