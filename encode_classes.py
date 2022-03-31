import numpy as Numpy

# Convert input list of selected categories into a one hot encoded vector.
def encode_classes(class_range):
    
    # Print out the integer mapping to each class.
    #print('Mapping of class to integer based on index of dataset.')
    #print('0: plane\n1: cabinet\n2: car\n3: chair\n4: lamp\n5: couch\n6: table\n7: watercraft')
    
    # Mapping of class index integer for dataset:
    # 0: plane
    # 1: cabinet
    # 2: car
    # 3: chair
    # 4: lamp
    # 5: couch
    # 6: table
    # 7: watercraft
    
    # Split the string into individual classes using the delimiter.
    class_list = class_range.split(',')
    
    # Convert each entered class into lowercase.
    for index in range(len(class_list)):
        class_list[index] = class_list[index].lower()
    print('User specified multiclass classes:', class_list)

    # Dictionary to specify all possible classes.
    # Key for each entry represents the class index from the training and test datasets.
    
    # Mapping: Input index -> Output index.
    # Chair: 0 > 3
    # Table: 1 > 6
    # Couch: 2 > 5
    # Cabinet: 3 > 1
    # Lamp: 4 > 4
    # Car: 5 > 2
    # Plane: 6 > 0
    # Watercraft: 7 > 7
    
    class_dict = {3: 'chair', 6: 'table', 5: 'couch', 1: 'cabinet', 4: 'lamp', 2: 'car', 0: 'plane', 7: 'watercraft'}
    
    # List to hold all the one hot encoded arrays.
    master_list = []

    # For each user specified class.
    for category in class_list:
    
        # Retrieve the dictionary key corresponding to the class.
        index = list(class_dict.keys())[list(class_dict.values()).index(category)]
        
        # Append the chosen class index to the master list.
        master_list.append(index)

    # Convert the entire master list into an array.
    master_list = Numpy.array(master_list)
    #print('Integer encoded list:', master_list)
    return master_list

# This file cannot run as a standalone program.
if __name__ == "__main__":
    pass
