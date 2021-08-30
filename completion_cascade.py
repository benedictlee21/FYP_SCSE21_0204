




"""
completion cascaded test split
- downsample from the correcponding GT
- able to select a certain class or all classes
- able to support multi process
- argument can take in starting point and number of points
- can save and append to a file for the log
- in phase II, maybe try if DDP can do it or not.

"""

import os
import sys
import torch
import torchvision.utils as vutils

from utils.dgp_utils import *
from utils.treegan_utils import *
from dgp import DGP
import os.path as osp
import time
# from loss import *
from ChamferDistancePytorch.chamfer_python import distChamfer

import glob
import h5py
from plots.completion_pcd_plot import draw_any_set


### configs
# prepare arguments and save in config
# parser = utils.prepare_parser()
# parser = utils.add_dgp_parser(parser)
# import pdb; pdb.set_trace()
parser = prepare_parser_2()
# print(parser)
parser = add_dgp_parser(parser)
# parser = add_example_parser(parser)
config = vars(parser.parse_args())
opt = parser.parse_args()
opt.device = torch.device('cuda:'+str(opt.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(opt.device)
# opt.if_cuda_chamfer = if_cuda_chamfer # TODO
opt.if_cuda_chamfer = False
 # NOTE: vary DEGREE
if opt.degrees_opt  == 'default':
    opt.DEGREE = [1,  2,  2,  2,  2,  2, 64]
elif opt.degrees_opt  == 'opt_1':
    opt.DEGREE = [1,  2,  4, 16,  4,  2,  2]
elif opt.degrees_opt  == 'opt_2':
    opt.DEGREE = [1, 32,  4,  2,  2,  2,  2]
elif opt.degrees_opt  == 'opt_3':
    opt.DEGREE = [1,  4,  4,  4,  4,  4,  2]
elif opt.degrees_opt  == 'opt_4':
    opt.DEGREE = [1,  4,  4,  8,  4,  2,  2]
else:
    opt.DEGREE = [] # will report error
print(config)


### load data, as cpu_tensor
root_dir = '/mnt/lustre/share/zhangjunzhe'
cascade_pathanme = os.path.join(root_dir,'cascade','test_data.h5')

data = h5py.File(cascade_pathanme, 'r')
gt = torch.from_numpy(data['complete_pcds'][()])
partial = torch.from_numpy(data['incomplete_pcds'][()])
labels = data['labels'][()]

# index corresponds to the cls id in cascaded
cat_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']

# opt.class_choice = 'chair'
cat_id = cat_list.index(opt.class_choice)

index_ls = np.where(labels==cat_id)[0].tolist()

## init DGP model, model loaded during init
dgp = DGP(config,opt)

if opt.start_point == 0 and opt.run_len == 0:
    print('not implemented scenario!!!')

print('opt.start_point,run_len',opt.start_point,opt.run_len)
for i in range(opt.start_point, opt.start_point+opt.run_len):
    index = index_ls[i]
    target = partial[index].cuda()
    origin = gt[index].cuda()
    tic = time.time()
    dgp.reset_G()
    
    if opt.given_partial:
        dgp.set_target(origin=origin,target=target, category=None)
        dgp.downsample_target(downsample=opt.downsample, n_bins=opt.target_resol_bins)
    else:
        dgp.set_target(origin=origin, category=None)

    dgp.select_z(select_y=False)
    dgp.run()
    toc = time.time()
    print(index,'done, ',int(toc-tic),'s')

    if True:
        pcd_list = dgp.pcs_checkpoints
        flag_list = dgp.flags
        output_dir = '/mnt/lustre/zhangjunzhe/pcd_lib/saved_samples/invert_topnet_scan2_visual'
        output_stem = 'train_checkpoints_'+str(index)
        draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(2,6))
import pdb; pdb.set_trace()

