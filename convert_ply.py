

from plyfile import PlyData, PlyElement
import os
import numpy as np
import sys
import glob

def read_ply_xyz(input_dir):
    """ read XYZ point cloud from filename PLY file """

    pathnames = glob.glob(input_dir + '/*')
    ply_files = []
    count = 0

    for filepath in pathnames:
        if 'fake-z' in filepath:
            ply_files.append(filepath)
            count += 1
    print('Number of ply files: ', count)

    for PLY_file in ply_files:
        filename = os.path.basename(PLY_file)

        with open(PLY_file, 'rb') as file_in:
            plydata = PlyData.read(file_in)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            vertices[:,0] = plydata['vertex'].data['x']
            vertices[:,1] = plydata['vertex'].data['y']
            vertices[:,2] = plydata['vertex'].data['z']

        with open(filename + '.txt','w') as file_out:
            for i in vertices:
                file_out.write(str(i))
        print(filename, ' processed.')

if __name__ == "__main__":

    input_dir = sys.argv[1]
    read_ply_xyz(input_dir)
