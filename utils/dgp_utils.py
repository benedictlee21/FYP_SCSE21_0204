import argparse
# import logging

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.common_utils import *
# import dataset

# from .biggan_utils import CenterCropLongEdge

# Arguments for DGP
def add_dgp_parser(parser):
    parser.add_argument(
        '--dist', action='store_true', default=False,
        help='Train with distributed implementation (default: %(default)s)')
    parser.add_argument(
        '--port', type=str, default='12345',
        help='Port id for distributed training (default: %(default)s)')
    parser.add_argument(
        '--exp_path', type=str, default='',
        help='Experiment path (default: %(default)s)')
    parser.add_argument(
        '--root_dir', type=str, default='',
        help='Root path of dataset (default: %(default)s)')
    parser.add_argument(
        '--list_file', type=str, default='',
        help='List file of the dataset (default: %(default)s)')
    parser.add_argument(
        '--resolution', type=int, default=256,
        help='Resolution to resize the input image (default: %(default)s)')
    parser.add_argument(
        '--random_G', action='store_true', default=False,
        help='Use randomly initialized generator? (default: %(default)s)')
    # parser.add_argument(
    #     '--update_G', action='store_true', default=False,
    #     help='Finetune Generator? (default: %(default)s)')
    parser.add_argument('--update_G',default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--update_embed', action='store_true', default=False,
        help='Finetune class embedding? (default: %(default)s)')
    parser.add_argument(
        '--save_G', action='store_true', default=False,
        help='Save fine-tuned generator and latent vector? (default: %(default)s)')
    parser.add_argument(
        '--ftr_type', type=str, default='Discriminator',
        choices=['Discriminator', 'perceptual'],
        help='Feature loss type, choose from Discriminator and VGG (default: %(default)s)')
    parser.add_argument(
        '--ftr_num', type=int, default=[3], nargs='+',
        help='Number of features to computer feature loss (default: %(default)s)')
    parser.add_argument(
        '--ft_num', type=int, default=[2], nargs='+',
        help='Number of parameter groups to finetune (default: %(default)s)')
    parser.add_argument(
        '--print_interval', type=int, default=100, nargs='+',
        help='Number of iterations to print training loss (default: %(default)s)')
    parser.add_argument(
        '--save_interval', type=int, default=None, nargs='+',
        help='Number of iterations to save image')
    parser.add_argument(
        '--lr_ratio', type=float, default=[1.0, 1.0, 1.0, 1.0], nargs='+',
        help='Decreasing ratio for learning rate in blocks (default: %(default)s)')
    parser.add_argument(
        '--w_D_loss', type=float, default=[0.1], nargs='+',
        help='Discriminator feature loss weight (default: %(default)s)')
    parser.add_argument(
        '--w_nll', type=float, default=0.001,
        help='Weight for the negative log-likelihood loss (default: %(default)s)')
    parser.add_argument(
        '--w_mse', type=float, default=[0.1], nargs='+',
        help='MSE loss weight (default: %(default)s)')
    parser.add_argument(
        '--select_num', type=int, default=500,
        help='Number of image pool to select from (default: %(default)s)')
    parser.add_argument(
        '--sample_std', type=float, default=1.0,
        help='Std of the gaussian distribution used for sampling (default: %(default)s)')
    parser.add_argument(
        '--iterations', type=int, default=[200, 200, 200, 200], nargs='+',
        help='Training iterations for all stages')
    parser.add_argument(
        '--G_lrs', type=float, default=[1e-6, 2e-5, 1e-5, 1e-6], nargs='+',
        help='Learning rate steps of Generator')
    parser.add_argument(
        '--z_lrs', type=float, default=[1e-1, 1e-3, 1e-5, 1e-6], nargs='+',
        help='Learning rate steps of latent code z')
    parser.add_argument(
        '--warm_up', type=int, default=0,
        help='Number of warmup iterations (default: %(default)s)')
    parser.add_argument(
        '--use_in', type=str2bool, default=[False, False, False, False], nargs='+',
        help='Whether to use instance normalization in generator')
    parser.add_argument(
        '--stop_mse', type=float, default=0.0,
        help='MSE threshold for stopping training (default: %(default)s)')
    parser.add_argument(
        '--stop_ftr', type=float, default=0.0,
        help='Feature loss threshold for stopping training (default: %(default)s)')
    parser.add_argument('--save_inversion_path',default='',help='dir to save generated point clouds')   
    parser.add_argument('--use_emd_loss',default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dgp_mode', type=str, default='reconstruct',
        help='DGP mode (default: %(default)s)')
    parser.add_argument(
        '--hole_radius',type=float, default=0.5,
        help='radius of the single hole, ball hole')
    parser.add_argument(
        '--hole_k',type=int, default=500,
        help='k of knn ball hole')
    parser.add_argument(
        '--hole_n',type=int, default=1,
        help='n of knn hole or ball hole')
    parser.add_argument(
        '--n_bins',type=int, default=1,
        help='n of knn hole or ball hole')
    parser.add_argument(
        '--max_bins',type=int, default=16,
        help='n of knn hole or ball hole')
    parser.add_argument(
        '--target_resol_bins',type=int, default=1,
        help='n of knn hole or ball hole')
    parser.add_argument(
        '--cd_option',type=str, default='standard',
        help='if standard or single-sided CD')
    parser.add_argument('--init_by_ftr_loss',default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--increase_n_bins',default=False, type=lambda x: (str(x).lower() == 'true'))    
    parser.add_argument('--given_partial',default=True, type=lambda x: (str(x).lower() == 'false'))    
    parser.add_argument('--downsample',default=True, type=lambda x: (str(x).lower() == 'false'))    
    parser.add_argument('--start_point',type=int,default=0,help='for distributed')
    parser.add_argument('--run_len',type=int,default=0,help='for distributed')
    return parser