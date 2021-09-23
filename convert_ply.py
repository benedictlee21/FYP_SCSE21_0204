

from plyfile import PlyData, PlyElement
import os
import numpy as np
import sys
import glob

def read_ply_xyz(input_dir):
    """ read XYZ point cloud from filename PLY file """

    pathnames = glob.glob(input_dir + '/*')
    ply_files = []
    directory_list = []
    #count = 0

    for subpath in pathnames:
        directory_list.append(subpath)
        subdirectory = os.path.basename(subpath)
        one_shape = glob.glob(input_dir + subdirectory + '/*')

        for filepath in one_shape:
            if '.ply' in filepath:
                ply_files.append(filepath)
                #count += 1
    #print('Number of ply files: ', count)
    #print('Number of all ply files: ', len(ply_files))
    #print('Number of subdirectory paths: ', len(directory_list))
    file_count = 0
    folder_count = 0

    for each in ply_files:
        filename = os.path.basename(each)
        filename_no_ext = os.path.splitext(filename)[0]
        filename_ply_ext = directory_list[folder_count] + '/' + filename_no_ext + '.txt'
        file_count += 1
        #print('Total files: ', file_count)

        with open(each, 'rb') as file_in:
            plydata = PlyData.read(file_in)
            number_vertices = plydata['vertex'].count
            vertices = np.zeros(shape = [number_vertices, 3], dtype = np.float32)
            vertices[:,0] = plydata['vertex'].data['x']
            vertices[:,1] = plydata['vertex'].data['y']
            vertices[:,2] = plydata['vertex'].data['z']

        with open(filename_ply_ext,'w') as file_out:
            for coordinates in vertices:
                file_out.write(str(coordinates[0]) + ';' + str(coordinates[1]) + ';' + str(coordinates[2]) + '\n')
        print('Processed: ', filename_ply_ext)

        if file_count % 11 == 0:
            folder_count += 1
            print('Folder count: ', folder_count)

if __name__ == "__main__":

    input_dir = sys.argv[1]
    read_ply_xyz(input_dir)
