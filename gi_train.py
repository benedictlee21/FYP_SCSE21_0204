import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from collections import OrderedDict

import torch
import torchvision.utils as vutils
from data.dataset_benchmark import BenchmarkDataset
# import utils
from utils.dgp_utils import *
from utils.treegan_utils import *
# from models import DGP
from dgp import DGP
import os.path as osp
import time
from loss import *
from ChamferDistancePytorch.chamfer_python import distChamfer
# above chamfer is from: https://github.com/ThibaultGROUEIX/ChamferDistancePytorch

import glob
import h5py
try: 
    import chamfer
    if_cuda_chamfer = True
except:
    if_cuda_chamfer = False
### below is EMD from https://github.com/daerduoCarey/PyTorchEMD 

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
# print(opt)
# utils.dgp_update_config(config)
# set random seed  # TODO
# utils.seed_rng(config['seed'])


### pick pcs that are not seen by GAN
inversion_unseen_pcs = False
# inversion_unseen_pcs = False
n_inversion = 100
data_1k = BenchmarkDataset(root=config['dataset_path'], npoints=config['point_num'], uniform=None, class_choice=config['class_choice'],n_samples_train=config['n_samples_train'])
datapath_1k = data_1k.get_datapath()
print(len(datapath_1k))
data_all = BenchmarkDataset(root=config['dataset_path'], npoints=config['point_num'], uniform=None, class_choice=config['class_choice'],n_samples_train=0)
datapath_all = data_all.get_datapath()
print(len(datapath_all))
# print(datapath_all[0])
datapath_test = sorted(list(set(datapath_all)-set(datapath_1k)))
print(len(datapath_test))
np.random.seed(0)
pathnames_to_invert = []
if inversion_unseen_pcs:
    perm_list = np.random.permutation(len(datapath_test))
    for i in range(n_inversion):
        pathnames_to_invert.append(datapath_test[perm_list[i]][1])
else:
    perm_list = np.random.permutation(len(datapath_1k))
    for i in range(n_inversion):
        pathnames_to_invert.append(datapath_1k[perm_list[i]][1])

print('len and sample of pcs',len(pathnames_to_invert),pathnames_to_invert[0])         
# time.sleep(10)

# # TODO num workers to change back to 4
# # pin_memory no effect
# dataLoader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=16)
# for _iter, data in enumerate(dataLoader):
#     pcs, pathnames = data
#     pathnames = sorted(pathnames)
#     print(pathnames[0])
#     for name in pathnames:
#         # splits = 
#         print(osp.basename(name))
# pcs_dir = '/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0/03001627/points/'
# basenames = [
#     '37b6df64a97a5c29369151623ac3890b.pts',
#     '4217f2ce7ecc286689c81af3a850d0ca.pts',
#     '490cc736a3ffc2a9c8687ff9b0b4e4ac.pts',
#     'a4ebefeb5cc26f7378a371eb63283bfc.pts',
#     'b6457a76f24de9f67aa6f8353fce2005.pts',
#     'be0c5a0e91c99e804e1a714ee619465a.pts',
#     'c658d1006595797e301c83e03ee59295.pts',
#     'ca5d7ee5cc56f989490ad276cd2af3a4.pts',
#     'd3f393615178dd1fa770dbd79b470bea.pts',
#     'e6f37dff25ec4ca4f815ebdb2df45512.pts']
# # print(basenames)
# pcs_list = []
# for basename in basenames:
#     pathname = osp.join(pcs_dir,basename)
#     point_set = np.loadtxt(pathname).astype(np.float32)
#     perm_list = np.random.permutation(point_set.shape[0])
#     choice = perm_list[:2048]
#     point_set = point_set[choice]
#     # print('shape of point_set',point_set.shape)
#     point_set = torch.from_numpy(point_set)
#     pcs_list.append(point_set)

pcs_list = []
np.random.seed(0)
i = 0
for pathname in pathnames_to_invert:
    point_set = np.loadtxt(pathname).astype(np.float32)
    perm_list = np.random.permutation(point_set.shape[0])
    choice = perm_list[:2048]
    point_set = point_set[choice]
    # print('shape of point_set',point_set.shape)
    point_set = torch.from_numpy(point_set).to(opt.device)
    pcs_list.append(point_set)
    # if i < 10:
        # print(os.path.basename(pathname))
    # i+=1


### Section: if check against gt
# check_against_gt = True
# if check_against_gt:
#     pcs_list2 = []
#     np.random.seed(1)
#     for pathname in pathnames_to_invert:
#         point_set = np.loadtxt(pathname).astype(np.float32)
#         perm_list = np.random.permutation(point_set.shape[0])
#         choice = perm_list[:2048]
#         point_set = point_set[choice]
#         # print('shape of point_set',point_set.shape)
#         point_set = torch.from_numpy(point_set).to(opt.device)
#         pcs_list2.append(point_set)
#     cd_list = []
#     emd_list = []
   
#     pcs1 = torch.stack(pcs_list)
#     pcs2 = torch.stack(pcs_list2)
#     dist1, dist2, _, _ =  distChamfer(pcs1, pcs2)
#     print('dist1 and dist2', dist1.mean(), dist2.mean())
    # import pdb; pdb.set_trace()
    # cd_fn = ChamferLoss()
    # emd_fn = EMDLoss()
    # for i in range(len(pcs_list)):
    #     a = pcs_list[i].unsqueeze(0)
    #     b = pcs_list2[i].unsqueeze(0)
    #     cd_dist = np.asscalar(cd_fn(a,b).detach().cpu().numpy())
    #     emd_dist = np.asscalar(emd_fn(a,b).detach().cpu().numpy())
    #     cd_list.append(cd_dist)
    #     emd_list.append(emd_dist)
    # print('check against gt')
    # print('cd mean:', np.mean(np.array(cd_list)), '  var:', np.var(np.array(cd_list)))
    # print('emd mean:', np.mean(np.array(emd_list)), '  var:', np.var(np.array(emd_list)))


### init DGP model
dgp = DGP(config,opt)
category = None # TODO used for conditional GAN

print('use_emd_loss',opt.use_emd_loss)


### Section: write basenames from the shapenet v0
# print('to write basenames')
# ### write file names
# file1 = open("basenames.txt","w")
# # file1.write("Hello \n") 
# # file1.writelines(L) 
# for pathname in pathnames_to_invert:
#     file1.write(os.path.basename(pathname)+'\n')
# file1.close()
# print('saved basenames')


### Section: inversion on partial scan topnet data
# based on the top 100 data in above 

if opt.given_partial:
    file1 = open("./plots/basenames.txt","r")
    v0_basenames = []
    for line in file1:
        stem, ext = line.rstrip('\n').split('.')
        v0_basenames.append(stem+'.h5')
    v1_data_path = '/mnt/lustre/share/zhangjunzhe/topnet/shapenet'
    cls_offset = '/03001627'
    basename = '6c25ec1178e9bab6e545858398955dd1.h5'
    v1_train_pathnames = glob.glob(v1_data_path+'/train/partial'+cls_offset+'/*')
    # v1_val_pathnames = glob.glob(v1_data_path+'/val/partial'+cls_offset+'/*')
    # v1_test_pathnames = glob.glob(v1_data_path+'/test/partial'+cls_offset+'/*')
    # import pdb; pdb.set_trace()
    v1_train_basenames = [os.path.basename(pathname) for pathname in v1_train_pathnames]
    # v1_val_basenames = 
    # v1_test_basenames = 

    ### construct basename list that are from v0_basenames, and v1_train_basenames
    intersect_basenames = [ itm for itm in v0_basenames if itm in v1_train_basenames]
    print('size of intersect basenames',len(intersect_basenames))
    print(intersect_basenames[68])
    # for basename in v0_basenames:


    ### construct a list of dict, with value : cuda tensor for partial and gt
    dir_gt = '/mnt/lustre/share/zhangjunzhe/topnet/shapenet/train/gt'+cls_offset
    dir_partial = '/mnt/lustre/share/zhangjunzhe/topnet/shapenet/train/partial'+cls_offset
    pcs_list = []
    for basename in intersect_basenames:
        this_shape = {}
        pathname_gt = os.path.join(dir_gt,basename)
        pathname_partial = os.path.join(dir_partial,basename)
        with h5py.File(pathname_partial, 'r') as f:
            pcd_partial = np.array(f['data'])
        with h5py.File(pathname_gt, 'r') as f:
            pcd_gt = np.array(f['data'])
        this_shape['gt'] = torch.from_numpy(pcd_gt).type(torch.float32).to(opt.device)
        this_shape['partial'] = torch.from_numpy(pcd_partial).type(torch.float32).to(opt.device)
        pcs_list.append(this_shape)

print('pcs_list[0] type',type(pcs_list[0]))

### Section: verify masked target
# for i in range(len(pcs_list)):
#     if i != 68:
#         continue
#     partial = pcs_list[i]['partial'].unsqueeze(0)
#     gt = pcs_list[i]['gt'].unsqueeze(0)
#     pcd_list = [partial, gt]
#     flag_list = ['target', 'gt']
#     for n in [4, 8, 16, 32, 64]:
#         dgp.reset_G()
#         dgp.set_target(origin=pcs_list[i]['gt'],target=pcs_list[i]['partial'], category=None)

#         # draw gt_regen target
#         new_gt = dgp.pre_process(gt, n_bins=n)
#         pcd_list.append(new_gt)
#         flag_list.append('n='+str(n))
    
#     output_dir = '/mnt/lustre/zhangjunzhe/pcd_lib/saved_samples/invert_topnet_scan2_visual'
#     output_stem = 'trial_gt_masked'
#     draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(2,4))
        

#### Section: check one sided CD
# p2f = []
# f2p = []
# p2f1 = []
# f2p1 = []
# for n in [4, 8, 16, 32, 64]:
#     print('***************** n =',n)
#     for i in range(len(pcs_list)):
#         # if i != 68:
#             # continue
#         partial = pcs_list[i]['partial'].unsqueeze(0)
#         gt = pcs_list[i]['gt'].unsqueeze(0)
        
#         # a, b, _, _ = distChamfer(partial, gt)
#         # print(a.mean(), '     ', b.mean())

#         dgp.reset_G()
#         dgp.set_target(origin=pcs_list[i]['gt'],target=pcs_list[i]['partial'], category=None)
#         a1, b1, _, _ = distChamfer(dgp.target, dgp.target_origin)
#         # print('--------before downsample',a1.mean().item(), '     ', b1.mean().item())
#         dgp.downsample_target(downsample=True,n_bins=n)
#         # remove 0 from target
#         target = []
#         for point in dgp.target[0]:
#             if torch.all(torch.eq(point, torch.tensor([0,0,0]).type(torch.float32).cuda())):
#                 continue
#             target.append(point)
#         dgp.target = torch.stack(target).unsqueeze(0)
            
#         a, b, _, _ = distChamfer(dgp.target, dgp.target_origin)
#         p2f.append(a.mean().item())
#         f2p.append(b.mean().item())
#         p2f1.append(a1.mean().item())
#         f2p1.append(b1.mean().item())
#     print('p2f vs f2p before downsample',np.array(p2f1).mean(),np.array(f2p1).mean())
#     print('p2f vs f2p after downsample ',np.array(p2f).mean(),np.array(f2p).mean())

print('init_by_ftr_loss',opt.init_by_ftr_loss)
print('target_resol_bins', opt.target_resol_bins)
### train pcd
for i in range(len(pcs_list)):
    if i != 68:
        continue
    # target_pcd = pcs_list[0].to(opt.device)
    tic = time.time()
    dgp.reset_G()
    
    if opt.given_partial:
        dgp.set_target(origin=pcs_list[i]['gt'],target=pcs_list[i]['partial'], category=None)
        dgp.downsample_target(downsample=opt.downsample, n_bins=opt.target_resol_bins)
    else:
        dgp.set_target(origin=pcs_list[i], category=None)

   

    dgp.select_z(select_y=False)
    dgp.run()
    toc = time.time()
    print(i,'done, ',int(toc-tic),'s')

    # import pdb; pdb.set_trace()
    # draw checkpoints
    pcd_list = dgp.pcs_checkpoints
    flag_list = dgp.flags
    output_dir = '/mnt/lustre/zhangjunzhe/pcd_lib/saved_samples/invert_topnet_scan2_visual'
    output_stem = 'train_checkpoints'
    draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(2,6))

    ### debuging here
    # gt_np = dgp.target_origin[0].cpu().numpy()
    # gt_map = dgp.pre_process(dgp.target_origin, process_target=True)[0].cpu().numpy()
    # target_map_np = dgp.pre_process(dgp.target, process_target=True)[0].cpu().numpy()
    # target_np = dgp.target[0].cpu().numpy()
    # prefix = 'ref'
    
    # np.savetxt(osp.join(opt.save_inversion_path,prefix+str(1)+'_x.txt'), gt_map, fmt = "%f;%f;%f")  
    # np.savetxt(osp.join(opt.save_inversion_path,prefix+str(1)+'_xmap.txt'), target_map_np, fmt = "%f;%f;%f")  
    # np.savetxt(osp.join(opt.save_inversion_path,prefix+str(1)+'_target.txt'), target_np, fmt = "%f;%f;%f")  
    # np.savetxt(osp.join(opt.save_inversion_path,prefix+str(1)+'_gt.txt'), gt_np, fmt = "%f;%f;%f")  
    # import pdb; pdb.set_trace()


    
# print(dgp.loss_log)
print('number of examples',len(dgp.loss_log))
ftr_loss = []
cd = []
emd = []
nll = []
for loss in dgp.loss_log:
    ftr_loss.append(loss['ftr_loss'])
    cd.append(loss['cd'])
    emd.append(loss['emd'])
    nll.append(loss['nll'])

print('cd mean:', np.mean(np.array(cd)), '  var:', np.var(np.array(cd)))
print('emd mean:', np.mean(np.array(emd)), '  var:', np.var(np.array(emd)))
print('ftr_loss mean:', np.mean(np.array(ftr_loss)), '  var:', np.var(np.array(ftr_loss)))
print('nll mean:', np.mean(np.array(nll)), '  var:', np.var(np.array(nll)))