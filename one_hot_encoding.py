import numpy as Numpy

# Convert input list of selected categories into a one hot encoded vector.
def one_hot_encode_classes(class_range):
    
    # Split the string into individual classes using the delimiter.
    class_list = class_range.split(',')
    
    # Convert each entered class into lowercase.
    for index in range(len(class_list)):
        class_list[index] = class_list[index].lower()
    print('User specified multiclass classes:', class_list)

    # Dictionary to specify all possible classes.
    class_dict = {0:'chair', 1: 'table', 2: 'couch', 3: 'cabinet', 4: 'lamp', 5: 'car', 6: 'plane', 7: 'watercraft'}
    
    # List to hold all the one hot encoded arrays.
    master_list = []

    # For each user specified class.
    for category in class_list:
    
        # Retrieve the dictionary key corresponding to the class.
        index = list(class_dict.keys())[list(class_dict.values()).index(category)]
        
        # Create an array of zeros the same length as the dictionary.
        temp_array = Numpy.array([0] * len(class_dict))
        
        # Set the index of the chosen class in the array to 1.
        temp_array[index] = 1
        
        # Append the array to the master list.
        master_list.append(temp_array)

    # Convert the entire master list into an array.
    master_list = Numpy.array(master_list)
    print('One hot encoded array:', master_list)
    return master_list

# This file cannot run as a standalone program.
if __name__ == "__main__":
    pass