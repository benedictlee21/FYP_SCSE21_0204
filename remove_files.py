import os
import os.path as osp
import glob

input_dir_base = '/mnt/lustre/zhangjunzhe/pcd_lib/model/knn6m'

# input_dirs = [input_dir_base+str(i) for i in range(29)]
input_dirs = [input_dir_base]
# mode = 'keep_00'
# mode = '<1500'
# mode = 'keep_00_50'
mode = 'keep_0'

for input_dir in input_dirs:
    pathnames = glob.glob(input_dir+"/*")
    for pathname in pathnames:  
        if mode == 'keep_00':
            if '00' not in pathname:
                # print(pathname)
                os.remove(pathname)
        if mode == 'keep_00_50':
            if '00' not in pathname and '50' not in pathname:
                os.remove(pathname)
        if mode == 'keep_0':
            if '0' not in pathname:
                # print(pathname)
                os.remove(pathname)
                # print('')
            # else:
                # print(pathname)