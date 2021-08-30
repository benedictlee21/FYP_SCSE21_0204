


"""
This script is to verify if cascade test.h5 has the same order as in pcn.

The script runs locally.
"""

# if local:
#     root_dir = 

msn_val_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/msn/val.list'
msn_train_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/msn/train.list'

shapenet_train_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/shapenet/train.list'
shapenet_val_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/shapenet/val.list'
shapenet_test_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/shapenet/test.list'

topnet_train_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/topnet/train.list'
topnet_val_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/topnet/val.list'
topnet_test_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/topnet/test.list'

pcn_train_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/pcn/train.list'
pcn_val_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/pcn/valid.list'
pcn_test_pathname = '/Users/zhangjunzhe/Desktop/pcd_datasets/pcn/test.list'

### and cascaded

msn_train = open(msn_train_pathname).readlines()
topnet_train = open(topnet_train_pathname).readlines()
shapenet_train = open(shapenet_train_pathname).readlines()
pcn_train = open(pcn_train_pathname).readlines()

msn_val = open(msn_val_pathname).readlines()
topnet_val = open(topnet_val_pathname).readlines()
shapenet_val = open(shapenet_val_pathname).readlines()
pcn_val = open(pcn_val_pathname).readlines()

# msn_train = open(msn_train_list_pathname).readlines()
topnet_test = open(topnet_test_pathname).readlines()
shapenet_test = open(shapenet_test_pathname).readlines()
pcn_test = open(pcn_test_pathname).readlines()

# topnet and shapenet should be the same one
print('train if same', msn_train == topnet_train, msn_train == shapenet_train, msn_train == pcn_train)
print('val if same', msn_val == topnet_val, msn_val == shapenet_val, msn_val == pcn_val)
print('test if same',topnet_test == shapenet_test, topnet_test == pcn_test)
print('topnet train, val, test size:', len(topnet_train), len(topnet_val), len(topnet_test))
print('pcn train, val, test size:', len(pcn_train), len(pcn_val), len(pcn_test))
# (Pdb) p topnet_test[:5]
# ['all/0000\n', 'all/0001\n', 'all/0002\n', 'all/0003\n', 'all/0004\n']
# (Pdb) p pcn_test[:5]
# ['03001627/2842701d388dcd3d534fa06200d07790\n', '03001627/6601d179e754149e2f710dc8afceb40e\n', '03001627/7d8e6b132c64d909b161f36d4e309050\n', '03001627/663b17baebd39695398b2b77aa0c22b\n', '03001627/25ad35439836dd7c8fc307d246c19849\n']

### verify the overlapped set
intersect_topnet_pcn = set.intersection(set(topnet_test),set(pcn_test))
# import pdb; pdb.set_trace()
# print(msn_train)


### draw topnet 10 for each

topnet_dir = '/Users/zhangjunzhe/Desktop/pcd_datasets/topnet/test/partial/all'
import glob
from completion_pcd_plot import *
import h5py
import torch
pathnames = sorted(glob.glob(topnet_dir+"/*"))
stem_ls = []
pcd_ls = []
print(len(pathnames))
for pathname in pathnames:
    # print(pathname)
    basename = os.path.basename(pathname)
    stem, txt = os.path.splitext(basename)
    # idx2pathname[int(stem)] = pathname
    # pathname = idx2pathname[i]
    f = h5py.File(pathname, 'r')
    pcd = torch.from_numpy(np.array(f['data']))
    f.close()
    stem_ls.append(stem)
    pcd_ls.append(pcd)

for i in range(120):
    flag_ls = stem_ls[i*10:(i+1)*10]
    pcd_ls2 = pcd_ls[i*10:(i+1)*10]
    output_dir = '/Users/zhangjunzhe/Downloads/pointsets/benchmark/drawings_test'
    output_stem = str(i*10)
    draw_any_set(flag_ls,pcd_ls2,output_dir,output_stem,layout=(2,5))
    print(output_stem)
    # break




