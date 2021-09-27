import sys
import os
import glob

# APPENDS 3 x 1024 ROWS OF ZEROS TO INPUT SHAPES FOR MPC RESULT EVALUATION.
def append_zeros(input_dir):
    pathnames = glob.glob(input_dir + '/*')
    total_processed = 0

    for subpath in pathnames:
        subdirectory = os.path.basename(subpath)
        one_shape = glob.glob(input_dir + subdirectory + '/*')

        for filepath in one_shape:
            if 'raw.txt' in filepath:

                with open(filepath, 'a') as file_in:
                    for i in range(1000):
                        file_in.write('0.0; 0.0; 0.0\n')

                print('Processed:', filepath)
                total_processed += 1

    print('Total files processed:', total_processed)

if __name__ == '__main__':

    input_dir = sys.argv[1]
    append_zeros(input_dir)
