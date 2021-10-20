import torch.utils.data as data
import os
import os.path
import torch
import numpy as np

class load_multiclass_dataset():

    def __init__(self, dataset_path, num_points = 5000, uniform = False, classification = True, classes_chosen = None, ratio = None):
        
        # Variables beginning with 'self' belong to the particular instance of this class.
        self.num_points = num_points
        self.dataset_path = dataset_path
        self.uniform = uniform
        self.classification = classification
        
        # Assign the number of points per point cloud for each class.
        # Using the zip function with takes in iterable arguments and returns an iterator.
        self.category_to_index = {one_class, points for one_class, points in zip(classes_chosen, ratio)}
        print('Category to index:', self.category_to_index)
        
        # Specify the location of the text file containing indexes to separate the 
        # different classes in the input shape dataset file.
        self.category_file = ./data/synsetoffset2category.txt
        self.categories = {}
        
        # Open the category file as read only and extract each key and value per entry row.
        with open(self.category_file, 'r') as cat_file:
            for row in cat_file:
            
                # Strip returns a new string without the specified characters.
                # Split divides a string into a list using a whitespace seperator by default.
                entry = row.strip().split()
                
                # Concatenate the next entry read in to the previous entry in the dictionary.
                self.categories[entry[0]] = entry[1]
                
        if classes_chosen is not None:
            
            # Assign the offset index in the input dataset to each class name in the dictionary
            # only if that class is one of the selected classes for multiclass completion.
            self.categories = {class_name, offset_index for class_name, offset_index in self.categories.items() if class_name in classes_chosen}
            
        print('Multiclass chosen categories with indexes:', self.categories)
            
        self.metadata = {}
        
        # Set the random seed to a fixed value for reproducible results.
        np.random.seed(0)
                