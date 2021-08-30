import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

from math import ceil

import time
import visdom
import numpy as np
import argparse

from data.dataset_benchmark import BenchmarkDataset
from datasets import ShapeNet_v0
# from model.gan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd
from layers.gcn import TreeGCN

class ConditionalGenerator_v0(nn.Module):
    '''
    conditional GAN, ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
    v0: only cat, do nothing
    '''
    def __init__(self, batch_size, features, degrees, support, n_classes, version=0):
        super(ConditionalGenerator_v0, self).__init__()
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        self.version = version
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        # self.pointcloud = None
        vertex_num = 1
        self.gcn = nn.Sequential()
        # NOTE: for the first layer, add n_classes
        features[0]+= n_classes
        # NOTE: v1 instead of directly cat and feed into the gcn. feed into a fc first
        if self.version == 1 or self.version == 3:
            self.fc = nn.Sequential(
                nn.Linear(features[0], features[0]),
                nn.LeakyReLU(negative_slope=0.2),
            )
        if self.version == 2 or self.version == 4:
            self.fc = nn.Sequential(
                nn.Linear(features[0], 256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(256, features[0]),
                nn.LeakyReLU(negative_slope=0.2),
            )

        for inx in range(self.layer_num):
            #jz NOTE last layer activation False
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])
    def forward(self, tree, labels):
        # shape of feat[i] (B,nodes,features)
        # [torch.Size([64, 1, 96]), torch.Size([64, 1, 256]), torch.Size([64, 2, 256]), torch.Size([64, 4, 256]), 
        # torch.Size([64, 8, 128]), torch.Size([64, 16, 128]), torch.Size([64, 32, 128]), torch.Size([64, 2048, 3])]
        # import pdb; pdb.set_trace()
        if self.version == 0:
            tree[0] = torch.cat((tree[0],labels),-1)
        else:
            tree[0] = self.fc(torch.cat((tree[0],labels),-1))
        
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]
       
        return self.pointcloud

class ConditionalDiscriminator_v0(nn.Module):
    def __init__(self, batch_size, features, n_classes,version=0):
        super(ConditionalDiscriminator_v0, self).__init__()
        # import pdb; pdb.set_trace()
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        self.version = version

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        #jz below code got problem, linearity, and final sigmoid,  
        #jz TODO final softmax/sigmoid needed?
        # self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
        #                                  nn.Linear(features[-1], features[-2]),
        #                                  nn.Linear(features[-2], features[-2]),
        #                                  nn.Linear(features[-2], 1))
        
        # follow the r-GAN discriminator, just not very sure if got leaky relu right before sigmoid.
        # jz NOTE below got Sigmoid function
        feat_dim = features[-1] + n_classes
        # NOTE: v1 instead of directly cat and feed into the gcn. feed into a fc first
        if self.version == 3:
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LeakyReLU(negative_slope=0.2),
            )
        if self.version == 4:
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, 1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(1024, feat_dim),
                nn.LeakyReLU(negative_slope=0.2),
            )

        self.final_layer = nn.Sequential(
                    nn.Linear(feat_dim, 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid())
    
    def forward(self, f, y):
        # y shape (B, n_classes)        
        # feat shape (B,3,2048)
        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
        # import pdb; pdb.set_trace()
        # feat shape (B,dimension,2048) --> out (B,dimension)
        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        
        # NOTE cat here
        if self.version == 3 or self.version == 4:
            out = torch.cat((out,y.squeeze(1)),-1)
            out = self.fc(out)
        else:
            out = torch.cat((out,y.squeeze(1)),-1)

        # out (B,1)
        out = self.final_layer(out) # (B, 1)
        # import pdb; pdb.set_trace()
        return out

class ConditionalTreeGAN():
    def __init__(self, opt):
        self.opt = opt
        # ------------------------------------------------Dataset---------------------------------------------- #
        #jz default unifrom=True
        # self.data = BenchmarkDataset(root=opt.dataset_path, npoints=opt.point_num, uniform=None, class_choice=opt.class_choice)
        # TODO to refine here
        class_choice = ['Airplane','Car','Chair','Table']
        ratio = [500] * 4
        self.data = ShapeNet_v0(root=opt.dataset_path, npoints=opt.point_num, uniform=None, class_choice=class_choice,ratio=ratio)
        # TODO num workers to change back to 4
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=16)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = ConditionalGenerator_v0(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support, n_classes=opt.n_classes, version=opt.version).to(opt.device)
        # import pdb; pdb.set_trace()
        #jz default features=0.5*opt.D_FEAT
        self.D = ConditionalDiscriminator_v0(batch_size=opt.batch_size, features=opt.D_FEAT, n_classes=opt.n_classes).to(opt.device)             
        #jz parallel
        # self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)

        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(0, 0.99))
        #jz TODO check if think can be speed up via multi-GPU
        self.GP = GradientPenalty(opt.lambdaGP, gamma=1, device=opt.device)
        print("Network prepared.")
        # import pdb; pdb.set_trace()
    
    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):
        
        loss_log = {'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())

        metric = {'FPD': []}
        epoch_log = 0
        if load_ckpt is not None:
            checkpoint = torch.load(load_ckpt, map_location=self.opt.device)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())

            metric['FPD'] = checkpoint['FPD']
            
            print("Checkpoint loaded.")
        
        for epoch in range(epoch_log, self.opt.epochs):
            epoch_g_loss = []
            epoch_d_loss = []
            epoch_time = time.time()
            for _iter, data in enumerate(self.dataLoader):
                # NOTE change opt.batchsize into labels.shape[0], due to the last batch
                # Start Time
                # if _iter < 270:
                #     continue
                # # TODO remove
                # if _iter > 10:
                #     break
                
                start_time = time.time()
                point, labels = data
                # TODO to work on treeGCN batch size issue.
                if labels.shape[0] != self.opt.batch_size:
                    continue
                point = point.to(self.opt.device)
                labels = labels.to(self.opt.device)
                labels_onehot = torch.FloatTensor(labels.shape[0], opt.n_classes).to(self.opt.device)
                labels_onehot.zero_()
                labels_onehot.scatter_(1, labels, 1)
                labels_onehot.unsqueeze_(1)
                # -------------------- Discriminator -------------------- #
                tic = time.time()
                for d_iter in range(self.opt.D_iter):
                    self.D.zero_grad()
                    # in tree-gan: normal distribution with mean 0, variance 1.
                    # in r-gan: mean 0 , sigma 0.2. link https://github.com/optas/latent_3d_points/blob/master/notebooks/train_raw_gan.ipynb
                    # in GCN GAN: sigma 0.2 https://github.com/diegovalsesia/GraphCNN-GAN/blob/master/gconv_up_aggr_code/main.py
                    z = torch.randn(labels.shape[0], 1, 96).to(self.opt.device)
                    gen_labels = torch.from_numpy(np.random.randint(0, opt.n_classes, labels.shape[0]).reshape(-1,1)).to(self.opt.device)
                    gen_labels_onehot = torch.FloatTensor(labels.shape[0], opt.n_classes).to(self.opt.device)
                    gen_labels_onehot.zero_()
                    gen_labels_onehot.scatter_(1, gen_labels, 1)
                    gen_labels_onehot.unsqueeze_(1)

                    # NOTE: type may not be compatible
                    # import pdb; pdb.set_trace()
                    # print('iter and  update d',_iter,d_iter)
                    tree = [z]
                    
                    with torch.no_grad():
                        fake_point = self.G(tree,gen_labels_onehot)         
                        
                    D_real = self.D(point,labels_onehot)
                    D_realm = D_real.mean()

                    D_fake = self.D(fake_point,gen_labels_onehot)
                    D_fakem = D_fake.mean()

                    # TODO try remove gp_loss and see how (tried, loss explode)
                    gp_loss = self.GP(self.D, point.data, fake_point.data, conditional=True, yreal=labels_onehot)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    self.optimizerD.step()

                loss_log['D_loss'].append(d_loss.item())   
                epoch_d_loss.append(d_loss.item())          
                toc = time.time()
                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()
                
                z = torch.randn(labels.shape[0], 1, 96).to(self.opt.device)
                gen_labels = torch.from_numpy(np.random.randint(0, opt.n_classes, labels.shape[0]).reshape(-1,1)).to(self.opt.device)
                gen_labels_onehot = torch.FloatTensor(labels.shape[0], opt.n_classes).to(self.opt.device)
                gen_labels_onehot.zero_()
                gen_labels_onehot.scatter_(1, gen_labels, 1)
                gen_labels_onehot.unsqueeze_(1)
                
                tree = [z]
                
                fake_point = self.G(tree,gen_labels_onehot)
                G_fake = self.D(fake_point,gen_labels_onehot)
                G_fakem = G_fake.mean()
                
                g_loss = -G_fakem
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


            # ---------------- Epoch everage loss   --------------- #
            d_loss_mean = np.array(epoch_d_loss).mean()
            g_loss_mean = np.array(epoch_g_loss).mean()
            
            print("[Epoch] ", "{:3}".format(epoch),
                "[ D_Loss ] ", "{: 7.3f}".format(d_loss_mean), 
                "[ G_Loss ] ", "{: 7.3f}".format(g_loss_mean), 
                "[ Time ] ", "{:.2f}s".format(time.time()-epoch_time))
            epoch_time = time.time()
            # ---------------- Frechet Pointcloud Distance --------------- #
            if epoch % 5 == 0 and not result_path == None:
                fake_pointclouds = torch.Tensor([])
                # jz, adjust for different batch size
                # TODO change back to 5000
                test_batch_num = int(2000/self.opt.batch_size)
                for i in range(test_batch_num): # For 5000 samples
                    # print(i)
                    z = torch.randn(self.opt.batch_size, 1, 96).to(self.opt.device)
                    gen_labels = torch.from_numpy(np.random.randint(0, opt.n_classes, self.opt.batch_size).reshape(-1,1)).to(self.opt.device)
                    gen_labels_onehot = torch.FloatTensor(self.opt.batch_size, opt.n_classes).to(self.opt.device)
                    gen_labels_onehot.zero_()
                    gen_labels_onehot.scatter_(1, gen_labels, 1)
                    gen_labels_onehot.unsqueeze_(1)
                    tree = [z]
                    with torch.no_grad():
                        sample = self.G(tree,gen_labels_onehot).cpu()
                    fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)

                fpd = calculate_fpd(fake_pointclouds, statistic_save_path=self.opt.FPD_path, batch_size=100, dims=1808, device=self.opt.device)
                metric['FPD'].append(fpd)
                print('-------------------------[{:4} Epoch] Frechet Pointcloud Distance <<< {:.2f} >>>'.format(epoch, fpd))

                class_name = opt.class_choice if opt.class_choice is not None else 'all'
                # TODO to enable save fake points
                # torch.save(fake_pointclouds, result_path+str(epoch)+'_'+class_name+'.pt')
                del fake_pointclouds

            # ---------------------- Save checkpoint --------------------- #
            class_name = opt.class_choice if opt.class_choice is not None else 'all'
            if epoch % 1 == 0 and not save_ckpt == None:
                torch.save({
                        'epoch': epoch,
                        'D_state_dict': self.D.state_dict(),
                        'G_state_dict': self.G.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                        'FPD': metric['FPD']
                }, save_ckpt+str(epoch)+'_'+class_name+'.pt')

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        # Dataset arguments
        self._parser.add_argument('--dataset_path', type=str, default='/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0', help='Dataset file path.')
        self._parser.add_argument('--class_choice', type=str, default='Chair', help='Select one class to generate. [Airplane, Chair, ...] (default:all_class)')
        #jz TODO batch size default 20
        self._parser.add_argument('--batch_size', type=int, default=64, help='Integer value for batch size.')
        self._parser.add_argument('--point_num', type=int, default=2048, help='Integer value for number of points.')

        # Training arguments
        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        #jz TODO epoch default 2000
        self._parser.add_argument('--epochs', type=int, default=2, help='Integer value for epochs.')
        self._parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
        #jz NOTE if do grid search, need to specify the ckpt_path and ckpt_save.
        self._parser.add_argument('--ckpt_path', type=str, default='./model/checkpoints_cgan_v0/', help='Checkpoint path.')
        self._parser.add_argument('--ckpt_save', type=str, default='tree_ckpt_', help='Checkpoint name to save.')
        self._parser.add_argument('--ckpt_load', type=str, help='Checkpoint name to load. (default:None)')
        self._parser.add_argument('--result_path', type=str, default='./model/generated_cgan_v0/', help='Generated results path.')
        self._parser.add_argument('--result_save', type=str, default='tree_pc_', help='Generated results name to save.')
        self._parser.add_argument('--visdom_port', type=int, default=8097, help='Visdom port number. (default:8097)')
        self._parser.add_argument('--visdom_color', type=int, default=4, help='Number of colors for visdom pointcloud visualization. (default:4)')

        # Network arguments
        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
        self._parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
        self._parser.add_argument('--DEGREE', type=int, default=[1,  2,   2,   2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
        #jz default is default=[3,  64,  128, 256, 512, 1024]
        # for D in r-GAN
        self._parser.add_argument('--D_FEAT', type=int, default=[3, 64,  128, 256, 256, 512], nargs='+', help='Features for discriminator.')
        # for D in source code
        # self._parser.add_argument('--D_FEAT', type=int, default=[3,  64,  128, 256, 512, 1024], nargs='+', help='Features for discriminator.')
        # Evaluation arguments
        self._parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')
        self._parser.add_argument('--n_classes',type=int, default=16)
        self._parser.add_argument('--version', type=int, default=0)

    def parser(self):
        return self._parser


if __name__ == '__main__':
    opt = Arguments().parser().parse_args()
    opt.device = torch.device('cuda:'+str(opt.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(opt.device)

    SAVE_CHECKPOINT = opt.ckpt_path + opt.ckpt_save if opt.ckpt_save is not None else None
    LOAD_CHECKPOINT = opt.ckpt_load if opt.ckpt_load is not None else None #NOTE: changed for more flexible loading
    RESULT_PATH = opt.result_path + opt.result_save

    model = ConditionalTreeGAN(opt)
    model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT, result_path=RESULT_PATH)