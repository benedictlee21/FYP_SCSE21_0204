import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_benchmark import BenchmarkDataset
from datasets import ShapeNet_v0
from model.gan_network import Discriminator, Generator, ConditionalGenerator_v0, ConditionalDiscriminator_v0
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd

from arguments import Arguments

import time
import visdom
import numpy as np
from loss import *
from metrics import *
import os.path as osp
import os

class TreeGAN():
    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #
        # NOTE: for cgan, use ShapeNet_v0 also
        #jz default unifrom=True
        if args.dataset == 'ShapeNet_v0':
            class_choice = ['Airplane','Car','Chair','Table']
            ratio = [args.ratio_base] * 4
            self.data = ShapeNet_v0(root=args.dataset_path, npoints=args.point_num, uniform=None, class_choice=class_choice,ratio=ratio)
        elif args.dataset == 'ShapeNet_v0_rGAN_Chair':
            self.data = ShapeNet_v0_rGAN_Chair()
        else:
            self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, uniform=None, class_choice=args.class_choice,n_samples_train=args.n_samples_train)
        # TODO num workers to change back to 4
        # pin_memory no effect
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        print('if conditional:',self.args.conditional)
        # import pdb; pdb.set_trace()
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        if self.args.conditional:
            self.G = ConditionalGenerator_v0(batch_size=args.batch_size, features=args.G_FEAT, degrees=args.DEGREE, support=args.support, n_classes=args.n_classes, version=args.cgan_version,args=self.args).to(args.device)

            self.D = ConditionalDiscriminator_v0(batch_size=args.batch_size, features=args.D_FEAT, n_classes=args.n_classes,version=args.cgan_version).to(args.device)
        else:
            self.G = Generator(batch_size=args.batch_size, features=args.G_FEAT, degrees=args.DEGREE, support=args.support,version=0,args=self.args).to(args.device)
            # import pdb; pdb.set_trace()
            #jz default features=0.5*args.D_FEAT
            self.D = Discriminator(batch_size=args.batch_size, features=args.D_FEAT).to(args.device)             
        print('network types',type(self.G),type(self.D))
        #jz parallel
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)

        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))
        #jz TODO check if think can be speed up via multi-GPU
        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
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
        color_num = self.args.visdom_color
        chunk_size = int(self.args.point_num / color_num)
        #jz TODO???
        colors = np.array([(227,0,27),(231,64,28),(237,120,15),(246,176,44),
                           (252,234,0),(224,221,128),(142,188,40),(18,126,68),
                           (63,174,0),(113,169,156),(164,194,184),(51,186,216),
                           (0,152,206),(16,68,151),(57,64,139),(96,72,132),
                           (172,113,161),(202,174,199),(145,35,132),(201,47,133),
                           (229,0,123),(225,106,112),(163,38,42),(128,128,128)])
        colors = colors[np.random.choice(len(colors), color_num, replace=False)]
        label = torch.stack([torch.ones(chunk_size).type(torch.LongTensor) * inx for inx in range(1,int(color_num)+1)], dim=0).view(-1)

        epoch_log = 0
        
        loss_log = {'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())

        metric = {'FPD': []}
        if load_ckpt is not None:
            checkpoint = torch.load(load_ckpt, map_location=self.args.device)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            try:
                self.G.load_state_dict(checkpoint['G_state_dict'])
            except: # NOTE temp fix for loading not para trained modeles #TODO
                self.G = Generator(batch_size=self.args.batch_size, features=self.args.G_FEAT, degrees=self.args.DEGREE, support=self.args.support,version=0,args=self.args).to(self.args.device)
                self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())

            metric['FPD'] = checkpoint['FPD']
            
            print("Checkpoint loaded.")

        for epoch in range(epoch_log, self.args.epochs):
            epoch_g_loss = []
            epoch_d_loss = []
            epoch_time = time.time()
            for _iter, data in enumerate(self.dataLoader):
                # TODO remove
                # if _iter > 20:
                #     break
                # Start Time
                start_time = time.time()
                point, labels = data
                # TODO to work on treeGCN batch size issue.
                # import pdb; pdb.set_trace()
                if labels.shape[0] != self.args.batch_size:
                    continue
                point = point.to(self.args.device)
                labels = labels.to(self.args.device)
                labels_onehot = torch.FloatTensor(labels.shape[0], args.n_classes).to(self.args.device)
                labels_onehot.zero_()
                labels_onehot.scatter_(1, labels, 1)
                labels_onehot.unsqueeze_(1)

                # -------------------- Discriminator -------------------- #
                tic = time.time()
                for d_iter in range(self.args.D_iter):
                    self.D.zero_grad()
                    # in tree-gan: normal distribution with mean 0, variance 1.
                    # in r-gan: mean 0 , sigma 0.2. link https://github.com/optas/latent_3d_points/blob/master/notebooks/train_raw_gan.ipynb
                    # in GCN GAN: sigma 0.2 https://github.com/diegovalsesia/GraphCNN-GAN/blob/master/gconv_up_aggr_code/main.py
                    z = torch.randn(labels.shape[0], 1, 96).to(self.args.device)
                    gen_labels = torch.from_numpy(np.random.randint(0, args.n_classes, labels.shape[0]).reshape(-1,1)).to(self.args.device)
                    gen_labels_onehot = torch.FloatTensor(labels.shape[0], args.n_classes).to(self.args.device)
                    gen_labels_onehot.zero_()
                    gen_labels_onehot.scatter_(1, gen_labels, 1)
                    gen_labels_onehot.unsqueeze_(1)
                    tree = [z]
                    
                    with torch.no_grad():
                        if self.args.conditional:
                            fake_point = self.G(tree,gen_labels_onehot) 
                            D_real, _ = self.D(point,labels_onehot)
                            D_fake, _ = self.D(fake_point,gen_labels_onehot)
                            # D_real, _ = self.D(point)   
                            # D_fake, _ = self.D(fake_point)  
                        else:
                            fake_point = self.G(tree)   
                            D_real, _ = self.D(point)   
                            D_fake, _ = self.D(fake_point)   
                    
                    D_realm = D_real.mean()
                    D_fakem = D_fake.mean()

                    gp_loss = self.GP(self.D, point.data, fake_point.data, conditional=self.args.conditional,yreal=labels_onehot)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    self.optimizerD.step()

                loss_log['D_loss'].append(d_loss.item())   
                epoch_d_loss.append(d_loss.item())          
                toc = time.time()
                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()
                
                z = torch.randn(labels.shape[0], 1, 96).to(self.args.device)
                gen_labels = torch.from_numpy(np.random.randint(0, args.n_classes, labels.shape[0]).reshape(-1,1)).to(self.args.device)
                gen_labels_onehot = torch.FloatTensor(labels.shape[0], args.n_classes).to(self.args.device)
                gen_labels_onehot.zero_()
                gen_labels_onehot.scatter_(1, gen_labels, 1)
                gen_labels_onehot.unsqueeze_(1)
                tree = [z]
                if self.args.conditional:
                    fake_point = self.G(tree,gen_labels_onehot)
                    G_fake, _ = self.D(fake_point,gen_labels_onehot)
                else:
                    fake_point = self.G(tree)
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

                loss_log['G_loss'].append(g_loss.item())
                epoch_g_loss.append(g_loss.item())
                tac = time.time()
                # --------------------- Visualization -------------------- #
                verbose = None
                if verbose is not None:
                    print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                        "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                        "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                        "[ Time ] ", "{:4.2f}s".format(time.time()-start_time),
                        "{:4.2f}s".format(toc-tic),
                        "{:4.2f}s".format(tac-toc))

                # jz TODO visdom is disabled
                # if _iter % 10 == 0:
                #     generated_point = self.G.getPointcloud()
                #     plot_X = np.stack([np.arange(len(loss_log[legend])) for legend in loss_legend], 1)
                #     plot_Y = np.stack([np.array(loss_log[legend]) for legend in loss_legend], 1)

                #     self.vis.line(X=plot_X, Y=plot_Y, win=1,
                #                   opts={'title': 'TreeGAN Loss', 'legend': loss_legend, 'xlabel': 'Iteration', 'ylabel': 'Loss'})

                #     self.vis.scatter(X=generated_point[:,torch.LongTensor([2,0,1])], Y=label, win=2,
                #                      opts={'title': "Generated Pointcloud", 'markersize': 2, 'markercolor': colors, 'webgl': True})

                #     if len(metric['FPD']) > 0:
                #         self.vis.line(X=np.arange(len(metric['FPD'])), Y=np.array(metric['FPD']), win=3, 
                #                       opts={'title': "Frechet Pointcloud Distance", 'legend': ["{} / FPD best : {:.6f}".format(np.argmin(metric['FPD']), np.min(metric['FPD']))]})

                #     print('Figures are saved.')
            # ---------------- Epoch everage loss   --------------- #
            d_loss_mean = np.array(epoch_d_loss).mean()
            g_loss_mean = np.array(epoch_g_loss).mean()
            
            print("[Epoch] ", "{:3}".format(epoch),
                "[ D_Loss ] ", "{: 7.6f}".format(d_loss_mean), 
                "[ G_Loss ] ", "{: 7.6f}".format(g_loss_mean), 
                "[ Time ] ", "{:.2f}s".format(time.time()-epoch_time))
            epoch_time = time.time()
            # ---------------- Frechet Pointcloud Distance --------------- #
            test_opt = True # TODO True to test
            if test_opt and epoch % self.args.eval_every_n_epoch == 0 and not result_path == None:
                fake_pointclouds = torch.Tensor([])
                # jz, adjust for different batch size
                test_batch_num = int(5000/self.args.batch_size)
                for i in range(test_batch_num): # For 5000 samples
                    z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
                    gen_labels = torch.from_numpy(np.random.randint(0, args.n_classes, self.args.batch_size).reshape(-1,1)).to(self.args.device)
                    gen_labels_onehot = torch.FloatTensor(self.args.batch_size, args.n_classes).to(self.args.device)
                    gen_labels_onehot.zero_()
                    gen_labels_onehot.scatter_(1, gen_labels, 1)
                    gen_labels_onehot.unsqueeze_(1)
                    tree = [z]
                    with torch.no_grad():
                        if self.args.conditional:
                            sample = self.G(tree,gen_labels_onehot).cpu()
                        else:
                            sample = self.G(tree).cpu()
                    fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)

                fpd = calculate_fpd(fake_pointclouds, statistic_save_path=self.args.FPD_path, batch_size=100, dims=1808, device=self.args.device)
                metric['FPD'].append(fpd)
                print('---------------------------[{:4} Epoch] Frechet Pointcloud Distance <<< {:.2f} >>>'.format(epoch, fpd))
                # compute var for uniformness test
                r_list = [0.01, 0.02, 0.05]
                n_seeds_list = [200] * 6
                pcs = fake_pointclouds[:1000]
                vars = []
                for radius, n_seeds in zip(r_list, n_seeds_list):
                    var = cal_patch_density_variance(pcs, radius=radius, n_seeds=n_seeds, batch_size=100, device=self.args.device)        
                    vars.append(var.cpu().numpy())
                print('----------------var at [0.01, 0.02, 0.05] ======', int(vars[0]),int(vars[1]),int(vars[2]))
                class_name = args.class_choice if args.class_choice is not None else 'all'
                # TODO
                # torch.save(fake_pointclouds, result_path+str(epoch)+'_'+class_name+'.pt')
                del fake_pointclouds

            # ---------------------- Save checkpoint --------------------- #
            if epoch % self.args.eval_every_n_epoch == 0 and not save_ckpt == None:
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

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    
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
    print(args)
    model = TreeGAN(args)
    model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT, result_path=RESULT_PATH)
    # print(args)
