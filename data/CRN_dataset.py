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
        
        # Value of string 'split' determines which '.h5' dataset file is used, either training or test.
        self.split = self.args.split
        
        print('CRN_dataset.py: __init__ - classes chosen:', classes_chosen)

        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')
        
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]
        
        category_ordered_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']
        print('Category ordered list:', category_ordered_list)

        # Import shapes from input dataset for multiclass.
        if classes_chosen is not None:
            self.category_id_list = []
            
            # For each class category chosen:
            for each_class in classes_chosen:
            
                # If the class is one of the valid categories.
                if each_class in category_ordered_list:
                
                    # Append the indexes of the chosen classes to a list.
                    # Convert the classes chosen to lowercase before comparison.
                    self.category_id_list.append(category_ordered_list.index(each_class.lower()))
            
            print('CRN_dataset multiclass category ID list by index:', self.category_id_list)
            
            # Master list of lists to hold all the indexes of all the classes for multiclass pretraining.
            self.index_list = []
            count = 0
            
            # Obtain the indexes of each class for all classes being used for multiclass pretraining.
            for each_class in self.category_id_list:
                self.one_class_index = np.array([i for (i, j) in enumerate(self.labels) if j == self.category_id_list[count]])
                
                # View the indexes for each individual class in the numpy array.
                print('Category index:', self.category_id_list[count])
                print('Shape indexes:', self.one_class_index)
                count += 1
                    
                # Need to reduce the training set size.
                # Take only 1000 out of 5750 sample shapes for training.
                self.one_class_index = self.one_class_index.tolist()
                
                # Get input argument for the number of samples per class to use for multiclass pretraining.
                self.one_class_index = random.sample(self.one_class_index, self.args.samples_per_class)
                
                # Append all the indexes from each class into a master list.
                for index in self.one_class_index:
                    self.index_list.append(index)
            
            # Shuffle the class indexes in the master list at random to ensure even model pretraining across all classes.
            # Set the random seed to obtain reproducible results.
            #np.random.seed(0)
            random.shuffle(self.index_list)
            #print('CRN multiclass shuffled index list:', self.index_list)
            self.index_list = np.array(self.index_list)
            print('Multiclass index list length:', len(self.index_list))
            
        # Import shapes from input dataset for single class.
        else:
            # Obtain a single category ID corresponding to the single class.
            category_id = category_ordered_list.index(self.class_choice.lower())
            #print('CRN_dataset single category ID:', cat_id)
        
            # Extract all the shapes from the input dataset that match the selected single class.
            self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == category_id ])
        
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
