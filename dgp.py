import os
import os.path as osp
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
# from PIL import Image
# from skimage import color
# from skimage.measure import compare_psnr, compare_ssim
from torch.autograd import Variable
from model.gan_network import Generator, Discriminator

from utils.common_utils import *
from loss import *
from evaluation.pointnet import *
import time
from ChamferDistancePytorch.chamfer_python import distChamfer

# import models
# import utils
# from models.downsampler import Downsampler


class DGP(object):

    def __init__(self, config, opt):
        self.rank, self.world_size = 0, 1
        if config['dist']:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.config = config
        self.opt = opt
        self.to_reset_mask = True # init seed for ball_hole or real_mask
        self.n_bins = opt.n_bins
        self.bins_list = [4, 8, 16, 32, 64]
        self.target_resol_bins = opt.target_resol_bins
        # self.mask_dict = {}
        self.mode = config['dgp_mode']
        self.update_G = config['update_G']
        self.update_embed = config['update_embed']
        self.iterations = config['iterations']
        self.ftr_num = config['ftr_num']
        self.ft_num = config['ft_num']
        self.lr_ratio = config['lr_ratio']
        self.G_lrs = config['G_lrs']
        self.z_lrs = config['z_lrs']
        self.use_in = config['use_in']
        self.select_num = config['select_num']
        self.factor = 2 if self.mode == 'hybrid' else 4  # Downsample factor

        self.pcs_cnt = 0
        self.loss_log = []
        # create model
        self.G = Generator(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support,version=0,args=self.opt).to(opt.device)
        # import pdb; pdb.set_trace()
        #jz default features=0.5*opt.D_FEAT
        self.D = Discriminator(batch_size=opt.batch_size, features=opt.D_FEAT).to(opt.device)      
        
        # to save intermediate variables
        self.flags = []
        self.pcs_checkpoints = []
        #jz parallel TODO to do here before distributed inversion
        # self.G = nn.DataParallel(self.G)
        # self.D = nn.DataParallel(self.D)

        # self.D = self.D.to(opt.device) 
        # TODO do not update G as of now
        # self.G.optim = torch.optim.Adam(
        #     [{'params': self.G.get_params(i, self.update_embed)}
        #         for i in range(len(self.G.blocks) + 1)],
        #     lr=config['G_lr'],
        #     betas=(config['G_B1'], config['G_B2']),
        #     weight_decay=0,
        #     eps=1e-8)
        # TODO settings, lr, etc!!!
        self.G.optim = torch.optim.Adam(self.G.parameters(),lr=self.opt.G_lrs[0],betas=(0,0.99))
        self.z = torch.zeros((1, 1, 96)).normal_().to(opt.device)
        self.z = Variable(self.z, requires_grad=True)
        self.z_optim = torch.optim.Adam([self.z], lr=self.opt.z_lrs[0], betas=(0,0.99))
        print('self.z at init',self.z.requires_grad)
        
        # import pdb; pdb.set_trace()
        # load weights
        # import pdb; pdb.set_trace()
        checkpoint = torch.load(opt.ckpt_load, map_location=self.opt.device)
        # TODO
        # pathname = '/mnt/lustre/zhangjunzhe/pcd_lib/model/temp/'+'test'+'.pt'
        # checkpoint = torch.load(pathname, map_location=opt.device)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        # try:
        #     self.G.load_state_dict(checkpoint['G_state_dict'])
        # except: # NOTE temp fix for loading not para trained modeles
        #     self.G = Generator(batch_size=self.opt.batch_size, features=self.opt.G_FEAT, degrees=self.opt.DEGREE, support=self.opt.support,version=0,args=self.opt).to(self.opt.device)
        #     self.G.load_state_dict(checkpoint['G_state_dict'])
        # try:
        #     self.D.load_state_dict(checkpoint['D_state_dict'])
        #     print('try load D')
        # except:
        #     print('except load D')
        #     self.D = Discriminator(batch_size=opt.batch_size, features=opt.D_FEAT).to(opt.device) 
        #     self.D.load_state_dict(checkpoint['D_state_dict'])
        # print('load D done')
        self.G.eval()
        if self.D is not None:
            self.D.eval()
        self.G_weight = deepcopy(self.G.state_dict())
    

        # prepare latent variable and optimizer
        # self._prepare_latent()
        # prepare learning rate scheduler # TODO
        self.G_scheduler = LRScheduler(self.G.optim, config['warm_up'])
        self.z_scheduler = LRScheduler(self.z_optim, config['warm_up'])

        # loss functions
        # self.mse = torch.nn.MSELoss()
        # TODO CD structual loss

        if config['ftr_type'] == 'Discriminator':
            self.ftr_net = self.D
            # TODO to make sure that D is sync when it is updated.
            self.criterion = DiscriminatorLoss()
        else:
            self.ftr_net = PointNetCls(k=16)
            PointNet_pretrained_path = './evaluation/cls_model_39.pth'
            self.ftr_net.load_state_dict(torch.load(PointNet_pretrained_path))
            # vgg = torchvision.models.vgg16(pretrained=True).cuda().eval()
            # self.ftr_net = models.subsequence(vgg.features, last_layer='20')
            self.criterion = PerceptLoss()
        # ChamferLoss
        if self.opt.if_cuda_chamfer:
            self.cd = ChamferLoss()
            self.emd = EMDLoss()

        print('init DGP done')
        # Downsampler for producing low-resolution image
        # self.downsampler = Downsampler(
        #     n_planes=3,
        #     factor=self.factor,
        #     kernel_type='lanczos2',
        #     phase=0.5,
        #     preserve_size=True).type(torch.cuda.FloatTensor)

    # def _prepare_latent(self):
    #     self.z = torch.zeros((1, 1, 96)).normal_().cuda()
    #     self.z = Variable(self.z, requires_grad=True)
    #     # TODO B1  and B2 are tuning knobs as well.
    #     # self.z_optim = torch.optim.Adam(
    #     #     [{'params': self.z, 'lr': self.z_lrs[0]}],
    #     #     # betas=(self.config['G_B1'], self.config['G_B2']),
    #     #     betas=(0,0.999),
    #     #     weight_decay=0,
    #     #     eps=1e-8
    #     # )

    #     self.z_optim = torch.optim.Adam(
    #         [{'params': self.z, 'lr': 0.001}],
    #         # betas=(self.config['G_B1'], self.config['G_B2']),
    #         betas=(0,0.999),
    #         weight_decay=0,
    #         eps=1e-8
    #     )
    #     self.y = torch.zeros(1).long().cuda()

    def reset_G(self):
        self.to_reset_mask = True # reset hole center for each shape
        self.pcs_cnt+=1
        self.G.load_state_dict(self.G_weight, strict=False) # TODO double confirm if deepcopy
        # self.G.reset_in_init() # TODO what does it mean??
        if self.config['random_G']:
            self.G.train()
        else:
            self.G.eval()
        self.flags = []
        self.pcs_checkpoints = []

    def random_G(self):
        self.G.init_weights()

    def set_target(self, origin=None, target=None, category=None):
        '''
        given original target, make partial pcd
        unsqueeze
        target is partial; origin is GT
        if target is not available, make our own partial by pre_process
        else: no need to do anything
        '''
        self.target_origin = origin.unsqueeze(0)
        if target is None:
            self.target = self.pre_process(self.target_origin,self.n_bins)
        else:
            self.target = target.unsqueeze(0)

    def run(self, save_interval=None):
        # print('to run dgp')
        # import pdb; pdb.set_trace()   
        # save_imgs = self.target.clone()
        # save_imgs2 = save_imgs.cpu().clone()
        loss_dict = {}
        curr_step = 0
        count = 0
        for stage, iteration in enumerate(self.iterations):
            # setup the number of features to use in discriminator
            # self.criterion.set_ftr_num(self.ftr_num[stage])
            # TODO reset every stage
            if self.opt.increase_n_bins:
                self.n_bins = min(self.opt.n_bins * (2**stage),self.opt.max_bins)
                print('self.n_bins',self.n_bins)

            for i in range(iteration):
                # if i < 2 or i > 196: 
                    # break
                    # print('centers',i,self.hole_centers)
                curr_step += 1
                # setup learning rate
                self.G_scheduler.update(curr_step, self.opt.G_lrs[stage])
                                        # self.ft_num[stage], self.lr_ratio[stage])
                self.z_scheduler.update(curr_step, self.opt.z_lrs[stage])

                self.z_optim.zero_grad()
                if self.update_G:
                    self.G.optim.zero_grad()
                             
                # tree = [self.z]
                x = self.G(self.z)
                # x = self.G(self.z, self.G.shared(self.y), use_in=self.use_in[stage])
                # apply degradation transform # TODO
                x_map = self.pre_process(x,self.n_bins)
                # import pdb; pdb.set_trace()   
                # calculate losses in the degradation space
                ftr_loss = self.criterion(self.ftr_net, x_map, self.target)
                if self.opt.if_cuda_chamfer:
                    cd_loss = self.cd(x_map,self.target)
                else:
                    if self.opt.cd_option == 'standard':
                        dist1, dist2 , _, _ = distChamfer(x_map, self.target)
                        cd_loss = dist1.mean() + dist2.mean()
                    else:
                        dist1, dist2 , _, _ = distChamfer(self.target,x)
                        cd_loss = dist1.mean() #p2f only
                if self.opt.use_emd_loss:
                    emd_loss = self.emd(x_map,self.target)

                # nll corresponds to a negative log-likelihood loss
                nll = self.z**2 / 2
                nll = nll.mean()
                # l1_loss = F.l1_loss(x_map, self.target)
                # loss = ftr_loss * self.config['w_D_loss'][stage] + \
                #     mse_loss * self.config['w_mse'][stage] + \
                #     nll * self.config['w_nll']
                w_emd = 0.1
                if self.opt.use_emd_loss:
                    loss = ftr_loss + nll * self.config['w_nll'] \
                        + cd_loss * 1 + emd_loss * w_emd
                else:
                    loss = ftr_loss + nll * self.config['w_nll'] \
                        + cd_loss * 1
                verbose = False
                if verbose:
                    if i < 5 or i > 190:
                        print(stage,'loss and emd at ',i,':',(loss-emd_loss*w_emd).detach().cpu().numpy(), emd_loss.detach().cpu().numpy())
                loss.backward()

                self.z_optim.step()
                # print(i, 'z   ',self.z[0,0,:5])
                # print(i, 'tree',tree[0][0,0,:5])
                
                if self.update_G:
                    self.G.optim.step()
                # print('ftr_loss',ftr_loss.detach().cpu().numpy(), type(ftr_loss.detach().cpu().numpy()))
        # compute cd and emd for full shapes, i.e. testing ; the other from training
            self.flags.append('stage_'+str(stage)+'x')
            self.pcs_checkpoints.append(x)
            self.flags.append('stage_'+str(stage)+'x_map')
            self.pcs_checkpoints.append(x_map)
            ## print test_cd
            dist1, dist2 , _, _ = distChamfer(x, self.target_origin)
            test_cd = dist1.mean() + dist2.mean()
            print('stage',stage, '--' , test_cd.item())
        if self.opt.if_cuda_chamfer:
            test_cd = self.cd(x, self.target_origin)
            test_emd = self.emd(x, self.target_origin)
        else:
            dist1, dist2 , _, _ = distChamfer(x, self.target_origin)
            test_cd = dist1.mean() + dist2.mean()
            test_emd = test_cd # NOTE: placeholder TODO
        
        loss_dict = {
            'ftr_loss': np.asscalar(ftr_loss.detach().cpu().numpy()),
            'nll': np.asscalar(nll.detach().cpu().numpy()),
            'cd': np.asscalar(test_cd.detach().cpu().numpy()),
            'emd': np.asscalar(test_emd.detach().cpu().numpy())
        }
        if self.opt.use_emd_loss:
            loss_dict['emd'] = np.asscalar(emd_loss.detach().cpu().numpy())
        self.loss_log.append(loss_dict)
        # import pdb; pdb.set_trace()
        self.save_pcs(self.G(self.z))
        print('save pcs done')

    def save_pcs(self, x, flag='fine_tuned'):
        """
        save x x_map, target, gt, only x is needed
        """
        if flag == 'fine_tuned':
            prefix = ''
            # import pdb; pdb.set_trace()
        else:
            prefix = 'pre'
        if not osp.isdir(self.opt.save_inversion_path):
            os.mkdir(self.opt.save_inversion_path)
        x_np = x[0].detach().cpu().numpy()
        np.savetxt(osp.join(self.opt.save_inversion_path,prefix+str(self.pcs_cnt)+'_x.txt'), x_np, fmt = "%f;%f;%f")  
        x_map = self.pre_process(x,self.n_bins)
        x_map_np = x_map[0].detach().cpu().numpy()
        target_np = self.target[0].detach().cpu().numpy()
        gt_np = self.target_origin[0].detach().cpu().numpy()
        
        np.savetxt(osp.join(self.opt.save_inversion_path,prefix+str(self.pcs_cnt)+'_xmap.txt'), x_map_np, fmt = "%f;%f;%f")  
        np.savetxt(osp.join(self.opt.save_inversion_path,prefix+str(self.pcs_cnt)+'_target.txt'), target_np, fmt = "%f;%f;%f")  
        np.savetxt(osp.join(self.opt.save_inversion_path,prefix+str(self.pcs_cnt)+'_gt.txt'), gt_np, fmt = "%f;%f;%f")  

    def select_z(self, select_y=False):
        
        tic = time.time()
        with torch.no_grad():
            if self.select_num == 0:
                self.z.zero_()
                return
            elif self.select_num == 1:
                self.z.normal_()
                return
            z_all, y_all, loss_all = [], [], []
            # if self.rank == 0:
                # print('Selecting z from {} samples'.format(self.select_num))
            # only use last 3 discriminator features to compare
            self.criterion.set_ftr_num(3)         
            for i in range(self.select_num):
                z = torch.randn(1, 1, 96).to(self.opt.device)
                # self.z.normal_(mean=0, std=1)
                # print('z and self.z shape',z.shape,self.z.shape)
                # tree = [z]
                with torch.no_grad():
                    x = self.G(z)
                # import pdb; pdb.set_trace()
                # NOTE: below will take much more time, need to redesign
                # x_map = self.pre_process(x,self.n_bins)
                # TODO x should be x_map
                if self.opt.init_by_ftr_loss:
                    ftr_loss = self.criterion(self.ftr_net, x, self.target)
                else:
                    dist1, dist2, _, _ = distChamfer(self.target, x)
                    ftr_loss = dist1.mean()

                z_all.append(z)
                loss_all.append(ftr_loss.detach().cpu().numpy())
            toc = time.time()
            # print('time spent in select z',int(toc-tic))
            loss_all = np.array(loss_all)
            idx = np.argmin(loss_all)
            # loss_all = torch.cat(loss_all)
            # idx = torch.argmin(loss_all)
            
            self.z.copy_(z_all[idx])
            if select_y:
                self.y.copy_(y_all[idx])
            self.criterion.set_ftr_num(self.ftr_num[0])
            
            ### save
            # print('time spent in select z',int(toc-tic))
            
            # print('self.z:',self.z.shape,self.z.requires_grad)
            # print('los max, mean, min',np.max(loss_all),np.mean(loss_all),np.min(loss_all))
            # print('min is which idx',idx, 'value:',loss_all[idx])
            # print('select z done')
            # print(self.z[0,0,:5])
            x = self.G(self.z)
            # x = self.target_origin # TODO debug only
            x_map = self.pre_process(x,self.n_bins)
            if self.opt.if_cuda_chamfer:
                cd_loss = self.cd(x_map,self.target)
            else:
                # print('self.target dtype', self.target.dtype)
                # print('x_map dtype', x_map.dtype)
                dist1, dist2 , _, _ = distChamfer(x_map, self.target)
                cd_loss = dist1.mean() + dist2.mean()            
            print('print init x_map and target cd',cd_loss.item())
            # print('select z done')
            # cd = self.cd
            self.flags.append('init x')
            self.pcs_checkpoints.append(x)
            self.flags.append('init x_map')
            self.pcs_checkpoints.append(x_map)
            # self.save_pcs(x,flag='init')
            return z_all[idx]

    def downsample_target(self,downsample,n_bins=32):
        """ TODO define n_bins"""
        # self.target = self.pre_process(self.target,n_bins=n_bins)
        # if downsample:
        #     self.target = self.pre_process(self.target_origin, n_bins=self.opt.target_resol_bins)
        self.flags.append('target')
        self.pcs_checkpoints.append(self.target.clone())
        
        self.target = self.pre_process(self.target_origin, n_bins, verbose=True)
        
        self.flags.append('target(down)')
        self.pcs_checkpoints.append(self.target)

    def pre_process(self, image, n_bins=0, verbose=False):
        ### TODO 
        # import pdb; pdb.set_trace()
        if self.mode == 'reconstruction':
            return image
        elif self.mode in ['ball_hole', 'knn_hole']:
            if self.to_reset_mask:
                self.hole_k = self.opt.hole_k
                self.hole_radius = self.opt.hole_radius
                self.hole_n = self.opt.hole_n
                seeds = farthest_point_sample(image, self.hole_n) # shape (B,hole_n)
                self.hole_centers = torch.stack([img[seed] for img, seed in zip(image,seeds)]) # (B, hole_n, 3)
                self.to_reset_mask = False
                
            flag_map = torch.ones(1,2048,1).to(self.opt.device)
            image_new = image.unsqueeze(2).repeat(1,1,self.hole_n,1)
            seeds_new = self.hole_centers.unsqueeze(1).repeat(1,2048,1,1)
            delta = image_new.add(-seeds_new) # (B, 2048, hole_n, 3)
            dist_mat = torch.norm(delta,dim=3)
            dist_new = dist_mat.transpose(1,2) # (B, hole_n, 2048)
            if self.mode == 'knn_hole':
                dist, idx = torch.topk(dist_new,self.hole_k,largest=False) # idx (B, hole_n, hole_k), dist (B, hole_n, hole_k)
            
            for i in range(self.hole_n):
                dist_per_hole = dist_new[:,i,:].unsqueeze(2)
                if self.mode == 'knn_hole':
                    threshold_dist = dist[:,i, -1]
                if self.mode == 'ball_hole': 
                    threshold_dist = self.hole_radius 
                flag_map[dist_per_hole <= threshold_dist] = 0
            image_map = torch.mul(image, flag_map)
            
            return image_map
        elif self.mode == 'topnet':

            if self.to_reset_mask:
                # print('reseting mask')
                self.mask_dict = {}
                for n in self.bins_list:
                    self.mask_dict[n] = {}
                    # NOTE: below self.target should be before downsampled
                     ### compute the [32]^3 {0,1} masks for the given target (partial pcd)
                    # it is a dict , the key is tuple, eg, (0,16,17)
                    pcd_new = self.target*n + n * 0.5
                    pcd_new = pcd_new.type(torch.int64)
                    ls_voxels = pcd_new.squeeze(0).tolist() # 2028 of sublists
                    tuple_voxels = [tuple(itm) for itm in ls_voxels]
                    for tuple_voxel in tuple_voxels:
                        if tuple_voxel not in self.mask_dict[n]:
                            self.mask_dict[n][tuple_voxel] = 1
                    # print(len(self.mask_dict[n].keys()))
                self.to_reset_mask = False
            # print('mask_dict n',n_bins,len(self.mask_dict[n_bins].keys())) #,self.mask_dict[4].keys())
            # compute the mask
            if n_bins == 0:
                return 0
            mask_tensor = torch.zeros(2048,1)
            pcd_new = image*n_bins + n_bins * 0.5
            pcd_new = pcd_new.type(torch.int64)
            ls_voxels = pcd_new.squeeze(0).tolist() # 2028 of sublists
            tuple_voxels = [tuple(itm) for itm in ls_voxels]
            for i in range(2048):
                tuple_voxel = tuple_voxels[i] # 0.01s
                if tuple_voxel in self.mask_dict[n_bins]:
                    mask_tensor[i] = 1
            mask_tensor = mask_tensor.unsqueeze(0).to(self.opt.device)
            image_map = torch.mul(image, mask_tensor)
            if verbose:
                print("n and 1's sum", n_bins, torch.sum(mask_tensor))
            # import pdb; pdb.set_trace()
            
            return image_map


    # def get_metrics(self, x):
    #     with torch.no_grad():
    #         l1_loss_origin = F.l1_loss(x, self.target_origin) / 2
    #         mse_loss_origin = self.mse(x, self.target_origin) / 4
    #         metrics = {
    #             'l1_loss_origin': l1_loss_origin,
    #             'mse_loss_origin': mse_loss_origin
    #         }
    #         # transfer to numpy array and scale to [0, 1]
    #         target_np = (self.target_origin.detach().cpu().numpy()[0] + 1) / 2
    #         x_np = (x.detach().cpu().numpy()[0] + 1) / 2
    #         target_np = np.transpose(target_np, (1, 2, 0))
    #         x_np = np.transpose(x_np, (1, 2, 0))
    #         if self.mode == 'colorization':
    #             # combine the 'ab' dim of x with the 'L' dim of target image
    #             x_lab = color.rgb2lab(x_np)
    #             target_lab = color.rgb2lab(target_np)
    #             x_lab[:, :, 0] = target_lab[:, :, 0]
    #             x_np = color.lab2rgb(x_lab)
    #             x = torch.Tensor(np.transpose(x_np, (2, 0, 1))) * 2 - 1
    #             x = x.unsqueeze(0)
    #         elif self.mode == 'inpainting':
    #             # only use the inpainted area to calculate ssim and psnr
    #             x_np = x_np[self.begin:self.end, self.begin:self.end, :]
    #             target_np = target_np[self.begin:self.end,
    #                                   self.begin:self.end, :]
    #         ssim = compare_ssim(target_np, x_np, multichannel=True)
    #         psnr = compare_psnr(target_np, x_np)
    #         metrics['psnr'] = torch.Tensor([psnr]).cuda()
    #         metrics['ssim'] = torch.Tensor([ssim]).cuda()
    #         return metrics, x

    # def jitter(self, x):
    #     save_imgs = x.clone().cpu()
    #     z_rand = self.z.clone()
    #     stds = [0.3, 0.5, 0.7]
    #     save_path = '%s/images/%s_jitter' % (self.config['exp_path'],
    #                                          self.img_name)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     with torch.no_grad():
    #         for std in stds:
    #             for i in range(30):
    #                 # add random noise to the latent vector
    #                 z_rand.normal_()
    #                 z = self.z + std * z_rand
    #                 x_jitter = self.G(z, self.G.shared(self.y))
    #                 utils.save_img(
    #                     x_jitter[0], '%s/std%.1f_%d.jpg' % (save_path, std, i))
    #                 save_imgs = torch.cat((save_imgs, x_jitter.cpu()), dim=0)

    #     torchvision.utils.save_image(
    #         save_imgs.float(),
    #         '%s/images_sheet/%s_jitters.jpg' %
    #         (self.config['exp_path'], self.img_name),
    #         nrow=int(save_imgs.size(0)**0.5),
    #         normalize=True)
