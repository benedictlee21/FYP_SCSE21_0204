import numpy

def perform_one_hot_encoding(class_range = None):
    
    # If the arguments for multiclass are specififed.
    if class_range is not None:
        # Read in list of classes that ShapeInversion should try to complete a partial input as.
        target_classes = class_range.split(',')
        print('Target classes:', target_classes)

        # Convert multiclass name inputs to lowercase.
        for index in range(len(target_classes)):
            target_classes[index] = target_classes[index].lower()

        # Dictionary to hold class name and integer indexes.
        # class_index_dict = {
            # "chair":0,
            # "table":1,
            # "couch":2,
            # "cabinet":3,
            # "lamp":4,
            # "car":5,
            # "plane":6,
            # "watercraft":7
        # }

        # Assign one hot indexes based on selected classes.
        # classes_chosen = [0,0,0,0,0,0,0,0]

        # for index in target_classes:
            # for entry in class_index_dict.keys():
                # if index == entry:
                    # classes_chosen[class_index_dict[entry]] = 1

        # # Convert the one hot encoding list into an array, representing the classes.
        # classes_chosen = numpy.array(classes_chosen)
        return target_classes

# Do not allow this python function to run as standalone.
if __name__ == "__main__":
    pass