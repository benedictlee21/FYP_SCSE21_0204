import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data.distributed import DistributedSampler
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")

input_size = 5
output_size = 2
batch_size = 30
data_size = 90

# 2） 配置每个进程的gpu
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size).to('cuda')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

dataset = RandomDataset(input_size, data_size)
# 3）使用DistributedSampler
rand_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         sampler=DistributedSampler(dataset))

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())
        return output
    
model = Model(input_size, output_size)

# 4) 封装之前要把模型移到对应的gpu
model.to(device)
    
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
   
for data in rand_loader:
    if torch.cuda.is_available():
        input_var = data
    else:
        input_var = data

    output = model(input_var)
    print("Outside: input size", input_var.size(), "output_size", output.size())


# """
# The goal of this file is to:
# * check number of points in the raw input
# * visualize the sampled points, say 100

# """

# ### Section Discriminator loss
# import torch
# import torch.nn as nn
# # from loss import *
# from model.gan_network import Generator, Discriminator


# a = torch.tensor([1, 2, 3])
# b = torch.tensor([1,2])
# print('ab',a)
# print(b)


# loss = 
# device = torch.device('cuda:'+str(opt.gpu) if torch.cuda.is_available())
# D_FEAT = [3, 64,  128, 256, 256, 512]
# D = Discriminator(batch_size=10, features=D_FEAT).cuda() 
# dloss = DiscriminatorLoss(ftr_num=0)
# # fake  = torch.randn

# # batchsize =  10
# # fake = torch.randn(batchsize,2048,3).cuda()
# # real = torch.randn(batchsize,2048,3).cuda()
# # # \
# # with torch.no_grad():
# #     _, real_feature = D(real.detach())
# # d, fake_feature = D(fake)    
# # D_penalty = F.l1_loss(fake_feature,real_feature)
# # print ('done')
# # _, loss = dloss(D,fake, real)
# fake = torch.randn(1,2048,3).cuda()
# real = torch.randn(1,2048,3).cuda()
# # \
# with torch.no_grad():
#     _, real_feature = D(real.detach())
# d, fake_feature = D(fake)    

# D_penalty = F.l1_loss(fake_feature,real_feature)
# print('feat and penalty shape',fake_feature.shape,D_penalty.shape,D_penalty)
# # D_penalty = F.l1_loss(fake_feature.squeeze,real_feature.squeeze)
# fake_feature = torch.squeeze(fake_feature,0)
# real_feature = torch.squeeze(real_feature,0)
# # print('feat and penalty shape',fake_feature.shape,D_penalty.shape,D_penalty)
# print ('done')
# _, loss = dloss(D,fake[0], real[0])
# print('called')


### Section test save models and save modules
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from data.dataset_benchmark import BenchmarkDataset
# from datasets import *
# from model.gan_network import Generator, Discriminator
# from model.gradient_penalty import GradientPenalty
# from evaluation.FPD import calculate_fpd

# from arguments import Arguments
# from utils.dgp_utils import *
# from utils.treegan_utils import *

# import time
# import visdom
# import numpy as np
# from loss import *
# from metrics import *
# import os.path as osp
# parser = prepare_parser_2()
# # print(parser)
# parser = add_dgp_parser(parser)
# # parser = add_example_parser(parser)
# config = vars(parser.parse_args())
# opt = parser.parse_args()
# opt.device = torch.device('cuda:'+str(opt.gpu) if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(opt.device)
#  # NOTE: vary DEGREE
# if opt.degrees_opt  == 'default':
#     opt.DEGREE = [1,  2,  2,  2,  2,  2, 64]
# elif opt.degrees_opt  == 'opt_1':
#     opt.DEGREE = [1,  2,  4, 16,  4,  2,  2]
# elif opt.degrees_opt  == 'opt_2':
#     opt.DEGREE = [1, 32,  4,  2,  2,  2,  2]
# elif opt.degrees_opt  == 'opt_3':
#     opt.DEGREE = [1,  4,  4,  4,  4,  4,  2]
# elif opt.degrees_opt  == 'opt_4':
#     opt.DEGREE = [1,  4,  4,  8,  4,  2,  2]
# else:
#     opt.DEGREE = [] # will report error
# print(config)
# data = BenchmarkDataset(root=opt.dataset_path, npoints=opt.point_num, uniform=None, class_choice=opt.class_choice,n_samples_train=100)
# dataLoader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True, pin_memory=True, num_workers=4)
# G = Generator(batch_size=50, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support,version=0,args=opt).to(opt.device)
#         # import pdb; pdb.set_trace()
#         #jz default features=0.5*opt.D_FEAT
# D = Discriminator(batch_size=50, features=opt.D_FEAT).to(opt.device)             
#         #jz parallel
# G = nn.DataParallel(G)
# D = nn.DataParallel(D)
# # for _iter, data in enumerate(dataLoader):
# # save
# class_name = opt.class_choice if opt.class_choice is not None else 'all'
# pathname = '/mnt/lustre/zhangjunzhe/pcd_lib/model/temp/'+'test'+'.pt'
# torch.save({
#             'epoch': 0,
#             'D_state_dict': D.module.state_dict(),
#             'G_state_dict': G.module.state_dict()
#             # 'D_loss': loss_log['D_loss'],
#             # 'G_loss': loss_log['G_loss'],
#             # 'FPD': metric['FPD']
#     }, pathname)

# # load
# checkpoint = torch.load(pathname, map_location=opt.device)
# G2 = Generator(batch_size=50, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support,version=0,args=opt).to(opt.device)
#         # import pdb; pdb.set_trace()
#         #jz default features=0.5*opt.D_FEAT
# D2 = Discriminator(batch_size=50, features=opt.D_FEAT).to(opt.device)    
# D2.load_state_dict(checkpoint['D_state_dict'])
# G2.load_state_dict(checkpoint['G_state_dict'])
# print('load done')

### Section: expansion penality
# import torch
# import sys
# sys.path.append("~/cuda_extension/MSN-Point-Cloud-Completion-master/expansion_penalty/")
# sys.path.append("~/cuda_extension/MSN-Point-Cloud-Completion-master/emd/")
# import emd_module as emd
# print(sys.path)
# EMD = emd.emdModule()

# eps = 0.005
# iters = 50
# n_seeds = 20
# batchsize = 64
# pcs = torch.randn(batchsize,2048,3).cuda()
# pcs2 = torch.randn(batchsize,2048,3).cuda()
# dist, _ = self.EMD(pcs, pcs2, eps, iters)
# emd1 = torch.sqrt(dist).mean(1)
# print(emd1)

# import time
# import numpy as np
# import torch
# from torch import nn
# from torch.autograd import Function
# import expansion_penalty

# # GPU tensors only
# class expansionPenaltyFunction(Function):
#     @staticmethod
#     def forward(ctx, xyz, primitive_size, alpha):
#         # 512 change to primitive_size TODO
#         assert(primitive_size <= 512)
#         batchsize, n, _ = xyz.size()
#         assert(n % primitive_size == 0)
#         xyz = xyz.contiguous().float().cuda()
#         dist = torch.zeros(batchsize, n, device='cuda').contiguous()
#         assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
#         neighbor = torch.zeros(batchsize, n * 512,  device='cuda', dtype=torch.int32).contiguous()
#         cost = torch.zeros(batchsize, n * 512, device='cuda').contiguous()
#         mean_mst_length = torch.zeros(batchsize, device='cuda').contiguous()
#         expansion_penalty.forward(xyz, primitive_size, assignment, dist, alpha, neighbor, cost, mean_mst_length)
#         ctx.save_for_backward(xyz, assignment)
#         return dist, assignment, mean_mst_length / (n / primitive_size)

#     @staticmethod
#     def backward(ctx, grad_dist, grad_idx, grad_mml):
#         xyz, assignment = ctx.saved_tensors
#         grad_dist = grad_dist.contiguous()
#         grad_xyz = torch.zeros(xyz.size(), device='cuda').contiguous()
#         expansion_penalty.backward(xyz, grad_xyz, grad_dist, assignment)
#         return grad_xyz, None, None

# class expansionPenaltyModule(nn.Module):
#     def __init__(self):
#         super(expansionPenaltyModule, self).__init__()

#     def forward(self, input, primitive_size, alpha):
#         return expansionPenaltyFunction.apply(input, primitive_size, alpha)

# def test_expansion_penalty():
#     x = torch.rand(128, 2048, 3).cuda()
#     print("Input_size: ", x.shape)
#     expansion = expansionPenaltyModule()
#     start_time = time.perf_counter()
#     dis, ass, mean_length = expansion(x, 64, 1.5)
#     import pdb; pdb.set_trace()
#     print("Runtime: %lfs" % (time.perf_counter() - start_time))

# test_expansion_penalty()

# Section: KNN (B, 2048,3) (B, 20, 3) seeds
# import torch
# from loss import *
# from sys import getsizeof

# n_seeds = 20
# batchsize = 64
# pcs = torch.randn(batchsize,2048,3)
# seeds = farthest_point_sample(pcs,n_seeds) # which gives index
# # print(seeds_idx)
# seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) # grad
# print('shape of pcs and seeds',pcs.shape,seeds.shape,seeds_value.shape)

# # dist = pcs - seeds
# # dist = pcs.add(-seeds)
# pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
# seeds_new = seeds_value.unsqueeze(1).repeat(1,2048,1,1)
# print(pcs_new.shape,seeds_new.shape)
# dist = pcs_new.add(-seeds_new)
# dist_value = torch.norm(dist,dim=3)
# print(dist.shape)
# print(dist_value.shape)
# dist_new = dist_value.transpose(1,2)
# print(dist_new.shape)
# short_dists, idx = torch.topk(dist_new, 10, dim=2, largest=False)
# print('return ans',idx.shape, short_dists.shape)
# # print('return ans',idx[0], short_dists[0])
# print('get size of',getsizeof(seeds_new),getsizeof(pcs))


### Section: simulation the density variance
# import torch
# import torch.nn.functional as F
# import numpy as np

# B = 5
# n_seeds = 30
# outlier_scale = 1
# n_outlier = 6
# n_repeat = 100
# var_a_ls = []
# var_a_n_ls = []
# var_a_o_ls = []
# var_a_o_n_ls = []
# for i in range(n_repeat):
#     a = torch.rand(B,n_seeds)
#     a_n = F.normalize(a)
#     var_a = torch.var(a,1).mean().numpy()
#     var_a_n = torch.var(a_n,1).mean().numpy()
    
#     # apply outlier
#     for i in range(n_outlier):
#         a[:,i] = 10**outlier_scale
#     var_a_o = torch.var(a,1).mean().numpy()
#     a_o_n = F.normalize(a)
#     var_a_o_n = torch.var(a_o_n,1).mean().numpy()
    
#     var_a_ls.append(var_a)
#     var_a_n_ls.append(var_a_n)
#     var_a_o_ls.append(var_a_o)
#     var_a_o_n_ls.append(var_a_o_n)

# print('var_a     mean:{:.3f}'.format(np.mean(var_a_ls)))
# print('var_a_n   mean:{:.3f}'.format(np.mean(var_a_n_ls)))
# print('var_a_o   mean:{:.3f}'.format(np.mean(var_a_o_ls)))
# print('var_a_o_n mean:{:.3f}'.format(np.mean(var_a_o_n_ls)))