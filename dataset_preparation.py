import os
import os.path
import torch
import numpy as np

from arguments import Arguments
from data.dataset_benchmark import BenchmarkDataset
import time

args = Arguments().parser().parse_args()

data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, uniform=False, class_choice=args.class_choice)
dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
tic = time.time()
for inx, pointcloud in enumerate(dataLoader):
    fn = data.datapath[inx]
    seg = np.loadtxt(fn[2]).astype(np.int64)
    choice = np.random.randint(0, len(seg), size=args.point_num)
    # import pdb; pdb.set_trace()
    directory = os.path.split(fn[3])[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(fn[3], choice)
    # break

toc = time.time()

print('preprcess done, time spent',int(toc-tic))