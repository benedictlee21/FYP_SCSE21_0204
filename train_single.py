import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_benchmark import BenchmarkDataset
from data.cascade_dataset import CascadeShapeNetv1, CascadeShapeNetv1_DDP
from datasets import *
from model.gan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd

from arguments import Arguments

import time
import visdom
import numpy as np
from loss import *
from metrics import *
import os.path as osp
from eval_GAN import checkpoint_eval
from apex import amp
import torch.distributed as dist

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

class TreeGAN():
    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #
        #jz default unifrom=True
        if args.dataset == 'ShapeNet_v0':
            class_choice = ['Airplane','Car','Chair','Table']
            ratio = [args.ratio_base] * 4
            self.data = ShapeNet_v0(root=args.dataset_path, npoints=args.point_num, uniform=None, class_choice=class_choice,ratio=ratio)
        elif args.dataset == 'ShapeNet_v0_rGAN_Chair':
            self.data = ShapeNet_v0_rGAN_Chair()
        elif args.dataset == 'cascade':
            self.data = CascadeShapeNetv1(class_choice=args.class_choice,n_samples=args.n_samples_train)
        elif args.dataset == 'cascade_ddp':
            self.data = CascadeShapeNetv1_DDP()
        else:
            self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, uniform=None, class_choice=args.class_choice,n_samples_train=args.n_samples_train)
        # TODO num workers to change back to 4
        # pin_memory no effect
        # import pdb; pdb.set_trace()
        if self.args.ddp_flag:
            torch.manual_seed(42) # TODO yet to verify if all model init the save value
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()

            # train_sampler = data.distributed.DistributedSampler(self.data, shuffle=True) # TODO shuffle looks got problem
            # self.dataLoader = data.DataLoader(
            #     self.data,
            #     batch_size=args.batch_size,
            #     sampler=train_sampler,
            #     drop_last=True,
            # )
            # torch.distributed.init_process_group(backend="nccl")
            # torch.cuda.set_device(self.args.local_rank)
            # torch.cuda.set_device(dist.get_rank())
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.data)
            self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, sampler=train_sampler)
        else:
            self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = Generator(batch_size=args.batch_size, features=args.G_FEAT, degrees=args.DEGREE, support=args.support,version=0,args=self.args).to(args.device)
        # import pdb; pdb.set_trace()
        #jz default features=0.5*args.D_FEAT
        self.D = Discriminator(batch_size=args.batch_size, features=args.D_FEAT).to(args.device)             
        if self.args.ddp_flag:
            # self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.args.local_rank])
            # print('G local rank',self.args.local_rank)
            # self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[self.args.local_rank])
            self.G = nn.parallel.DistributedDataParallel(
                self.G,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True #debug
            )

            self.D = nn.parallel.DistributedDataParallel(
                self.D,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True # debug
            )

        # else:
        # # self.G = nn.DataParallel(self.G)
        # # self.D = nn.DataParallel(self.D)

        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.pretrain_G_lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.pretrain_D_lr, betas=(0, 0.99))
        #jz TODO check if think can be speed up via multi-GPU
        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
        
        if self.args.apex_flag:
            ### TODO apex
            # model, optimizer = amp.initialize(model, optimizer)
            # self.G, self.D, self.optimizerG, self.optimizerD = amp.initialize(self.G, self.D, self.optimizerG, self.optimizerD)
            self.G, self.optimizerG = amp.initialize(self.G, self.optimizerG, opt_level="O1")
            # self.D, self.optimizerD = amp.initialize(self.D, self.optimizerD, opt_level="O1")
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        
        if self.args.expansion_penality:
            self.expansion = expansionPenaltyModule()
        if self.args.krepul_loss:
            self.krepul_loss = kNNRepulsionLoss(k=self.args.krepul_k,n_seeds=self.args.krepul_n_seeds,h=self.args.krepul_h)
        if self.args.knn_loss:
            self.knn_loss = kNNLoss(k=self.args.knn_k,n_seeds=self.args.knn_n_seeds)
        if self.args.patch_repulsion_loss:
            self.repul_loss = PatchRepulsionLoss(n_patches=32, n_sigma=self.args.n_sigma)
        if self.args.uniform_loss == 'custom_loss':
            self.uniform_loss = PatchDensityVariance(self.args.uniform_loss_radius,self.args.uniform_loss_n_seeds)
        elif self.args.uniform_loss == 'two_custom_loss':
            if self.args.radius_version == 'cus_los10':
                self.uniform_loss_1 = PatchDensityVariance(0.02, self.args.uniform_loss_n_seeds)
                self.uniform_loss_2 = PatchDensityVariance(0.05, self.args.uniform_loss_n_seeds)
                self.args.uniform_loss_scalar = [100, 0.02]
            elif self.args.radius_version == 'cus_los11':
                self.uniform_loss_1 = PatchDensityVariance(0.02, self.args.uniform_loss_n_seeds)
                self.uniform_loss_2 = PatchDensityVariance(0.05, self.args.uniform_loss_n_seeds)
                self.args.uniform_loss_scalar = [20, 0.1]
            elif self.args.radius_version == 'cus_los12':
                self.uniform_loss_1 = PatchDensityVariance(0.02, self.args.uniform_loss_n_seeds)
                self.uniform_loss_2 = PatchDensityVariance(0.05, self.args.uniform_loss_n_seeds)
                self.args.uniform_loss_scalar = [500, 0.004]
            elif self.args.radius_version == 'cus_los21':
                self.uniform_loss_1 = PatchDensityVariance(0.02, self.args.uniform_loss_n_seeds)
                self.uniform_loss_2 = PatchDensityVariance(0.05, self.args.uniform_loss_n_seeds)
                self.args.uniform_loss_scalar = [50, 0.05]
            elif self.args.radius_version == 'cus_los22':
                self.uniform_loss_1 = PatchDensityVariance(0.02, self.args.uniform_loss_n_seeds)
                self.uniform_loss_2 = PatchDensityVariance(0.05, self.args.uniform_loss_n_seeds)
                self.args.uniform_loss_scalar = [50, 0.1]
            elif self.args.radius_version == 'cus_los23':
                self.uniform_loss_1 = PatchDensityVariance(0.02, self.args.uniform_loss_n_seeds)
                self.uniform_loss_2 = PatchDensityVariance(0.05, self.args.uniform_loss_n_seeds)
                self.args.uniform_loss_scalar = [100, 0.1]
            elif self.args.radius_version == 'cus_los24':
                self.uniform_loss_1 = PatchDensityVariance(0.02, self.args.uniform_loss_n_seeds)
                self.uniform_loss_2 = PatchDensityVariance(0.05, self.args.uniform_loss_n_seeds)
                self.args.uniform_loss_scalar = [300, 0.1]
            elif self.args.radius_version == 'cus_los27':
                self.uniform_loss_1 = PatchDensityVariance(0.02, self.args.uniform_loss_n_seeds)
                self.uniform_loss_2 = PatchDensityVariance(0.05, self.args.uniform_loss_n_seeds)
                self.args.uniform_loss_scalar = [1000, 0.2]
            elif self.args.radius_version == 'cus_los28':
                self.uniform_loss_1 = PatchDensityVariance(0.02, self.args.uniform_loss_n_seeds)
                self.uniform_loss_2 = PatchDensityVariance(0.05, self.args.uniform_loss_n_seeds)
                self.args.uniform_loss_scalar = [2000, 0.2]
        print("Network prepared.")

        # ----------------------------------------------------------------------------------------------------- #

        # ---------------------------------------------Visualization------------------------------------------- #
        #jz TODO visdom
        # self.vis = visdom.Visdom(port=args.visdom_port)
        # assert self.vis.check_connection()
        # print("Visdom connected.")
        # ----------------------------------------------------------------------------------------------------- #

    def run(self, save_ckpt=None, load_ckpt=None, result_path=None): 
        # print('rank', get_rank())  
        # import pdb; pdb.set_trace()     
        color_num = self.args.visdom_color
        chunk_size = int(self.args.point_num / color_num)
        #jz TODO???

        epoch_log = 0
        
        loss_log = {'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())

        metric = {'FPD': []}
        if load_ckpt is not None:
            # import pdb; pdb.set_trace()
            checkpoint = torch.load(load_ckpt, map_location=self.args.device)
            # checkpoint = torch.load(load_ckpt).cuda() # yucunjun
            self.D.load_state_dict(checkpoint['D_state_dict'])
            # try:
                # self.G.load_state_dict(checkpoint['G_state_dict'])
            # except: # NOTE temp fix for loading not para trained modeles
                # self.G = Generator(batch_size=self.args.batch_size, features=self.args.G_FEAT, degrees=self.args.DEGREE, support=self.args.support,version=0,args=self.args).to(self.args.device)
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())

            metric['FPD'] = checkpoint['FPD']
            
            print("Checkpoint loaded.")
        # NOTE: parallel after loading
        if not self.args.ddp_flag:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        for epoch in range(epoch_log, self.args.epochs):
            epoch_g_loss = []
            epoch_d_loss = []
            epoch_time = time.time()
            # print('rank', get_rank())
            # import pdb; pdb.set_trace()
            for _iter, data in enumerate(self.dataLoader):
                # TODO remove
                # if _iter > 2:
                    # break
                # Start Time
                # if self.args.ddp_flag  and _iter == 0:
                    # and get_rank() == 0
                # print('rank', get_rank())
                # import pdb; pdb.set_trace()

                start_time = time.time()
                point, _ = data
                point = point.to(self.args.device)

                # -------------------- Discriminator -------------------- #
                tic = time.time()
                for d_iter in range(self.args.D_iter):
                    # self.D.zero_grad()
                    
                    # z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
                    # z = torch.randn(self.args.batch_size, 1, 96).to('cuda')
                    z = torch.randn(self.args.batch_size, 1, 96).cuda()
                    
                    with torch.no_grad():
                        fake_point = self.G(z)         
                    
                    D_real, _ = self.D(point)
                    D_realm = D_real.mean()

                    D_fake, _ = self.D(fake_point)
                    D_fakem = D_fake.mean()

                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    self.optimizerD.step()
                    if self.args.ddp_flag:
                        loss_dict = { 'd_loss_gp': d_loss_gp}
                        loss_reduced = reduce_loss_dict(loss_dict)

                loss_log['D_loss'].append(d_loss.item())   
                epoch_d_loss.append(d_loss.item())          
                toc = time.time()
                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()
                
                z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
                # tree = [z]
                
                fake_point = self.G(z)
                G_fake, _ = self.D(fake_point)
                G_fakem = G_fake.mean()
                # TODO make sure to overwrite it
                # TODO to refine the logic here
                g_loss = -G_fakem
                if self.args.expansion_penality:
                    dist, _, mean_mst_dis = self.expansion(fake_point,self.args.expan_primitive_size,self.args.expan_alpha)
                    #  = self.expansion(out1, self.num_points//self.n_primitives, 1.5)
                    expansion = torch.mean(dist)
                    g_loss = -G_fakem + self.args.expan_scalar * expansion
                    # import pdb; pdb.set_trace()
                if self.args.krepul_loss:
                    krepul_loss = self.krepul_loss(fake_point)
                    g_loss = -G_fakem + self.args.krepul_scalar * krepul_loss
                if self.args.knn_loss:
                    knn_loss = self.knn_loss(fake_point)
                    g_loss = -G_fakem + self.args.knn_scalar * knn_loss
                    # print(knn_loss)
                # print  
                if self.args.patch_repulsion_loss and self.args.uniform_loss == 'custom_loss':
                    repul_loss = self.repul_loss(fake_point)
                    uni_loss = self.uniform_loss(fake_point)
                    g_loss = -G_fakem + self.args.uniform_loss_scalar * uni_loss + repul_loss
                elif self.args.patch_repulsion_loss:
                    repul_loss = self.repul_loss(fake_point)
                    g_loss = -G_fakem + repul_loss
                elif self.args.uniform_loss == 'custom_loss':
                    uni_loss_value = self.uniform_loss(fake_point)
                    g_loss = -G_fakem + self.args.uniform_loss_scalar * uni_loss_value
                    # import pdb; pdb.set_trace()
                    # print('g loss and uni loss:',G_fakem.detach().cpu().numpy(),uni_loss_value.detach().cpu().numpy())
                elif self.args.uniform_loss == 'two_custom_loss':
                    uni_loss_1 = self.uniform_loss_1(fake_point)
                    uni_loss_2 = self.uniform_loss_2(fake_point)
                    g_loss = -G_fakem + self.args.uniform_loss_scalar[0] * uni_loss_1 + self.args.uniform_loss_scalar[1] * uni_loss_2
                elif self.args.uniform_loss == 'uniform_single_radius':
                    if (self.args.uniform_loss_warmup_till < epoch and self.args.uniform_loss_warmup_mode==0):
                        # if step and not yet to flag on uniform loss
                        continue
                    else:
                        # import pdb; pdb.set_trace()
                        uniform_scalar = self.args.uniform_loss_scalar
                        n_seeds = self.args.uniform_loss_n_seeds
                        radius = self.args.uniform_loss_radius
                        
                        seeds = farthest_point_sample(fake_point,n_seeds)
                        new_xyz = torch.stack([pc[seed] for pc, seed in zip(fake_point,seeds)])
                        density_mat = query_ball_point(radius, fake_point, new_xyz)
                        
                        # print(density_mat) #(B,n_seeds)
                        # softmax_op = torch.nn.Softmax()
                        # density_soft = softmax_op(density_mat)
                        # print(density_soft) #(B,n_seeds)
                        if self.args.uniform_loss_no_scale:
                            if self.args.uniform_loss_max:
                                # NOTE: actually it is max
                                den_var, _ = torch.max(density_mat,1)
                            else:
                                den_var = torch.var(density_mat,1)
                        else:
                            density_norm = torch.nn.functional.normalize(density_mat)
                            # density_max, _ = torch.max(density_mat,1) 
                            den_var = torch.var(density_norm,1)

                        if self.args.uniform_loss_offset != 0:
                            relu_offset = torch.Tensor([float(self.args.uniform_loss_offset)])
                            den_var = torch.nn.functional.relu(den_var-relu_offset)
                            
                        
                        # if self.args.uniform_loss_relu_var != 0:
                        #     relu_offset = torch
                        #     den_var  = torch.nn.functional()
                        # den_var2 = torch.var(density_mat,1)
                        # print(density_max)
                        # print('var soft',den_var)
                        # print('var mat',den_var2)
                        # print('var soft mean',den_var.mean().cpu().detach().numpy())
                        if epoch >= self.args.uniform_loss_warmup_till:
                            g_loss = -G_fakem + uniform_scalar*den_var.mean()
                        elif self.args.uniform_loss_warmup_mode==1 and self.args.uniform_loss_warmup_till > 0:
                            g_loss = -G_fakem + epoch/self.args.uniform_loss_warmup_till*uniform_scalar*den_var.mean()
                elif self.args.uniform_loss == 'uniform_multi_radius':
                    if self.args.radius_version == str(13):
                        uniform_scalar_list = [self.args.uniform_loss_scalar] * 6
                        n_seeds_list = [200, 100, 50, 20, 12, 10]
                        radius_list = [0.005, 0.01, 0.02, 0.05, 0.08, 0.1]
                        offset_list = [0] * 6
                    elif self.args.radius_version == str(14):
                        uniform_scalar_list = [self.args.uniform_loss_scalar] * 5
                        n_seeds_list = [200, 100, 50, 20, 12]
                        radius_list = [0.005, 0.01, 0.02, 0.05, 0.08]
                        offset_list = [0] * 5
                    elif self.args.radius_version == str(15):
                        uniform_scalar_list = [self.args.uniform_loss_scalar] * 5
                        n_seeds_list = [400, 200, 100, 40, 25]
                        radius_list = [0.005, 0.01, 0.02, 0.05, 0.08]
                        offset_list = [0] * 5
                    elif self.args.radius_version == str(23):
                        uniform_scalar_list = [0.5, 0.1, 0.05]
                        n_seeds_list = [200, 100, 40]
                        radius_list = [0.01, 0.02, 0.05]
                        offset_list = [0] * 3
                    elif self.args.radius_version == str(24):
                        uniform_scalar_list = [0.5, 0.1]
                        n_seeds_list = [200, 100]
                        radius_list = [0.01, 0.02]
                        offset_list = [0] * 2
                    elif self.args.radius_version == str(30):
                        uniform_scalar_list = [0.2, 0.1, 0.03, 0.01, 0.01]
                        radius_list = [0.01, 0.02, 0.05, 0.08, 0.1]
                        n_seeds_list = [200, 100, 50, 50, 50]
                        offset_list = [5, 10,40, 100, 150]
                    elif self.args.radius_version == str(31):
                        uniform_scalar_list = [0.2, 0.1, 0.03]
                        radius_list = [0.01, 0.02, 0.05]
                        n_seeds_list = [200, 100, 50]
                        offset_list = [5, 10,40]
                    elif self.args.radius_version == str(32):
                        uniform_scalar_list = [ 0.1, 0.03, 0.01]
                        radius_list = [ 0.02, 0.05, 0.08]
                        n_seeds_list = [100, 50, 50]
                        offset_list = [10,40, 100]
                    else:
                        print('not implemented condition!!!!!')
                    den_var_mean_sum = torch.Tensor([0]).to(self.args.device)
                    for n_seeds, radius, uniform_scalar, offset in zip(n_seeds_list, radius_list,uniform_scalar_list, offset_list):
                        seeds = farthest_point_sample(fake_point,n_seeds)
                        new_xyz = torch.stack([pc[seed] for pc, seed in zip(fake_point,seeds)])
                        density_mat = query_ball_point(radius, fake_point, new_xyz)
                        
                        if self.args.uniform_loss_no_scale:
                            if self.args.uniform_loss_max:
                                # NOTE: actually it is max
                                den_var, _ = torch.max(density_mat,1)
                            else:
                                den_var = torch.var(density_mat,1)
                        else:
                            density_norm = torch.nn.functional.normalize(density_mat)
                            # density_max, _ = torch.max(density_mat,1) 
                            den_var = torch.var(density_norm,1)

                        if self.args.uniform_loss_offset != 0:
                            relu_offset = torch.Tensor([float(offset)])
                            den_var = torch.nn.functional.relu(den_var-relu_offset)

                        den_var_mean_sum += den_var.mean()
                    g_loss = -G_fakem + uniform_scalar*den_var_mean_sum
                    # print('-----------------------G_fakem',-G_fakem)
                    # print('                       G_loss ',g_loss)
                
                ### add up loss TODO rearrange
                if self.args.expansion_penality and self.args.uniform_loss == 'custom_loss':
                    g_loss = -G_fakem + self.args.expan_scalar * expansion + self.args.uniform_loss_scalar * uni_loss_value
                elif self.args.expansion_penality and self.args.knn_loss:
                    g_loss = -G_fakem + self.args.expan_scalar * expansion + self.args.knn_scalar * knn_loss
                elif self.args.expansion_penality and self.args.krepul_loss:
                    g_loss = -G_fakem + self.args.expan_scalar * expansion + self.args.krepul_scalar * krepul_loss
                g_loss.backward()
                self.optimizerG.step()
                if self.args.ddp_flag:
                    loss_dict = { 'g_loss': g_loss}
                    loss_reduced = reduce_loss_dict(loss_dict)

                loss_log['G_loss'].append(g_loss.item())
                epoch_g_loss.append(g_loss.item())
                tac = time.time()
                # --------------------- Visualization -------------------- #
                # verbose = True
                verbose = False
                if verbose:
                    print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                        "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                        "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                        "[ Time ] ", "{:4.2f}s".format(time.time()-start_time),
                        "{:4.2f}s".format(toc-tic),
                        "{:4.2f}s".format(tac-toc))

            # ---------------- Epoch everage loss   --------------- #
            d_loss_mean = np.array(epoch_d_loss).mean()
            g_loss_mean = np.array(epoch_g_loss).mean()
            
            if get_rank() == 0:
                verbose_epoch = True
            else:
                verbose_epoch = False
            if verbose_epoch:
                print("[Epoch] ", "{:3}".format(epoch),
                    "[ D_Loss ] ", "{: 7.6f}".format(d_loss_mean), 
                    "[ G_Loss ] ", "{: 7.6f}".format(g_loss_mean), 
                    "[ Time ] ", "{:.2f}s".format(time.time()-epoch_time))
                epoch_time = time.time()

            ### call abstracted eval, which includes FPD, not uniformity yet
            if epoch % self.args.eval_every_n_epoch == 0 :
                checkpoint_eval(self.G, self.args.device, n_samples=5000, batch_size=100,conditional=False, FPD_path=self.args.FPD_path)


            #     pcs = fake_pointclouds[:1000]
            #     vars = []
            #     for radius, n_seeds in zip(r_list, n_seeds_list):
            #         var = cal_patch_density_variance(pcs, radius=radius, n_seeds=n_seeds, batch_size=100, device=self.args.device)        
            #         vars.append(var.cpu().numpy())
            #     print('----------------var at [0.01, 0.02, 0.05] ======', int(vars[0]),int(vars[1]),int(vars[2]))
            #     print('------------------------------------------------time in eval',int(time.time()-epoch_time))

            # ---------------------- Save checkpoint --------------------- #
            if epoch % 10 == 0 and not save_ckpt == None:
                class_name = args.class_choice if args.class_choice is not None else 'all'
                torch.save({
                        'epoch': epoch,
                        'D_state_dict': self.D.module.state_dict(),
                        'G_state_dict': self.G.module.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                        'FPD': metric['FPD']
                }, save_ckpt+str(epoch)+'_'+class_name+'.pt')

                # print('Checkpoint is saved.')
            
                
                    

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    # args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu') 
    # torch.cuda.set_device(args.device)
    # TODO for ddp
    args.device = "cuda"
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.ddp_flag


    # NOTE: vary DEGREE
    if args.degrees_opt  == 'default':
        args.DEGREE = [1,  2,  2,  2,  2,  2, 64]
    elif args.degrees_opt  == 'opt_1':
        args.DEGREE = [1,  2,  4, 16,  4,  2,  2]
    elif args.degrees_opt  == 'opt_2':
        args.DEGREE = [1, 32,  4,  2,  2,  2,  2]
    elif args.degrees_opt  == 'opt_3':
        args.DEGREE = [1,  4,  4,  4,  4,  4,  2]
    elif args.degrees_opt  == 'opt_4':
        args.DEGREE = [1,  4,  4,  8,  4,  2,  2]
    else:
        args.DEGREE = [] # will report error
    
    if not osp.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_load if args.ckpt_load is not None else None
    RESULT_PATH = args.result_path + args.result_save
    # print(args)
    model = TreeGAN(args)
    model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT, result_path=RESULT_PATH)
    # print(args)