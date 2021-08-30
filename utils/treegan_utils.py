from __future__ import print_function

import math
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.optimizer import Optimizer

def prepare_parser_2():
    # NOTE: from arguments.py used for GAN inversion stage
    print('called here, treegan_utils') # TODO
    usage = 'Parser for all scripts.'
    parser = ArgumentParser(description=usage)
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, default='/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0', help='Dataset file path.')
    parser.add_argument('--class_choice', type=str, default='Chair', help='Select one class to generate. [Airplane, Chair, ...] (default:all_class)')
    #jz TODO batch size default 20
    parser.add_argument('--batch_size', type=int, default=64, help='Integer value for batch size.')
    parser.add_argument('--point_num', type=int, default=2048, help='Integer value for number of points.')

    # Training arguments
    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
    #jz TODO epoch default 2000
    parser.add_argument('--epochs', type=int, default=2, help='Integer value for epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
    #jz NOTE if do grid search, need to specify the ckpt_path and ckpt_save.
    parser.add_argument('--ckpt_path', type=str, default='./model/checkpoints2/', help='Checkpoint path.')
    parser.add_argument('--ckpt_save', type=str, default='tree_ckpt_', help='Checkpoint name to save.')
    parser.add_argument('--ckpt_load', type=str, help='Checkpoint name to load. (default:None)')
    parser.add_argument('--result_path', type=str, default='./model/generated2/', help='Generated results path.')
    parser.add_argument('--result_save', type=str, default='tree_pc_', help='Generated results name to save.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Visdom port number. (default:8097)')
    parser.add_argument('--visdom_color', type=int, default=4, help='Number of colors for visdom pointcloud visualization. (default:4)')

    # Network arguments
    parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
    parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
    parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
    # NOTE: degree changed
    parser.add_argument('--degrees_opt', type=str, default='default', help='Upsample degrees for generator.')
    parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
    #jz default is default=[3,  64,  128, 256, 512, 1024]
    # for D in r-GAN
    parser.add_argument('--D_FEAT', type=int, default=[3, 64,  128, 256, 256, 512], nargs='+', help='Features for discriminator.')
    # for D in source code
    # parser.add_argument('--D_FEAT', type=int, default=[3,  64,  128, 256, 512, 1024], nargs='+', help='Features for discriminator.')
    # Evaluation arguments
    parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')
    parser.add_argument('--ratio_base', type=int, default=500, help='# to select for each cat')
    parser.add_argument('--dataset',type=str,default='BenchmarkDataset')
    parser.add_argument('--uniform_loss',type=str,default='None')
    parser.add_argument('--uniform_loss_scalar',type=float,default=20)
    parser.add_argument('--uniform_loss_radius',type=float,default=0.05)
    parser.add_argument('--uniform_loss_warmup_till',type=int,default=0) # if 0, no warmup, if 100, linear
    parser.add_argument('--uniform_loss_warmup_mode',type=int,default=1) # if 1, linear; if 0 step
    parser.add_argument('--radius_version', type=str,default='0')
    parser.add_argument('--uniform_loss_n_seeds', type=int,default=20)
    parser.add_argument('--uniform_loss_no_scale', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--uniform_loss_max', default=False, type=lambda x: (str(x).lower() == 'true'))    
    parser.add_argument('--uniform_loss_offset', type=int,default=0)
    parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--n_samples_train',type=int,default=0,help='# pcd to train')
    parser.add_argument('--eval_every_n_epoch',type=int,default=5)
    parser.add_argument('--patch_repulsion_loss',default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--n_sigma',type=float,default=1)
    parser.add_argument('--knn_loss',default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--knn_k',type=int,default=10)
    parser.add_argument('--knn_n_seeds',type=int,default=20)
    parser.add_argument('--knn_scalar',type=float,default=1)
    parser.add_argument('--krepul_loss',default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--krepul_k',type=int,default=10)
    parser.add_argument('--krepul_n_seeds',type=int,default=20)
    parser.add_argument('--krepul_scalar',type=float,default=1)
    parser.add_argument('--krepul_h',type=float,default=0.01)
    parser.add_argument('--expansion_penality',default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--expan_primitive_size',type=int,default=64)
    parser.add_argument('--expan_alpha',type=float,default=1.5)
    parser.add_argument('--expan_scalar',type=float,default=0.1)
    return parser