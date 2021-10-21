import torch.utils.data as data
import os
import os.path
import torch
import numpy as np

class load_multiclass_dataset(data.Dataset):

    def __init__(self, dataset_path, num_points = 5000, uniform = False, classification = True, classes_chosen = None, ratio = None):

        # Variables beginning with 'self' belong to the particular instance of this class.
        self.num_points = num_points
        self.dataset_path = dataset_path
        self.uniform = uniform
        self.classification = classification
        self.classes_chosen = classes_chosen

        # Assign the number of points per point cloud for each class.
        # Using the zip function with takes in iterable arguments and returns an iterator.
        self.category_to_index = {one_class:points for one_class, points in zip(self.classes_chosen, ratio)}
        print('Category to index:', self.category_to_index)

        # Specify the location of the text file containing indexes to separate the
        # different classes in the input shape dataset file.
        self.category_file = './data/synsetoffset2category.txt'
        self.categories = {}

        # Open the category file as read only and extract each key and value per entry row.
        with open(self.category_file, 'r') as cat_file:
            for row in cat_file:

                # Strip returns a new string without the specified characters.
                # Split divides a string into a list using a whitespace seperator by default.
                entry = row.strip().split()

                # Concatenate the next entry read in to the previous entry in the dictionary.
                self.categories[entry[0]] = entry[1]

        if self.classes_chosen is not None:
            self.selected_categories = {}
            
            # Assign the offset index in the input dataset to each class name in the dictionary
            # only if that class is one of the selected classes for multiclass completion.
            for class_name, offset_index in self.categories.items():
                
                # If one of the multiclass categories is chosen.
                if class_name in self.classes_chosen:
                    #print('class name:', class_name, ', offset index:', offset_index)
                    
                    # Add it to the dictionary of selected categories.
                    self.selected_categories[class_name] = offset_index
            
            #self.categories = {class_name:offset_index for class_name, offset_index in self.categories.items() if class_name in self.classes_chosen}

        print('Multiclass chosen categories with indexes:', self.selected_categories)
        self.metadata = {}

        # Set the random seed to a fixed value for reproducible results.
        np.random.seed(0)

        # For each chosen category for multiclass.
        for item in self.selected_categories:
            self.metadata[item] = []

            # Directory path for point cloud shapes for each category.
            shapes_directory = os.path.join(self.dataset_path, self.selected_categories[item], 'shapes')

            # Directory path for point cloud labels.
            label_directory = os.path.join(self.dataset_path, self.selected_categories[item], 'shape_label')

            # Directory for sampling.
            sampling_directory = os.path.join(self.dataset_path, self.selected_categories[item], 'sampling')

            # Sort the shapes in the directory of each category by name.
            sorted_shapes = sorted(os.listdir(shapes_directory))

            for entry in range(self.category_to_index[item]):
                token = os.path.splitext(os.path.basename(sorted_shapes[entry]))[0]

                # Append the paths for the point cloud directories to the metadata list.
                self.metadata[item].append(os.path.join(shapes_directory, token + '.pts'), os.path.join(label_directory, token + '.seg'), os.path.join(sampling_directory, token + '.sam'))
            self.datapath = []

            # For each category.
            for item in self.selected_categories:

                # For each shape entry in the metadata dictionary.
                for function in self.metadata[item]:

                    # Append the shape, label and sample to the datapath list.
                    self.datapath.append(item, function[0], function[1], function[2])

            self.classes = dict(zip(sorted(self.selected_categories), range(len(self.selected_categories))))
            print('Classes:', self.classes)
            print('Number of samples:', len(self.datapath))

    # Return the length of the object that is the instance of this class.
    def __len__(self):
        return len(self.datapath)
