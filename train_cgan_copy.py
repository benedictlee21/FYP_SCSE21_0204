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



class ConditionalTreeGAN():
    def __init__(self, opt):
        self.args = opt
        # ------------------------------------------------Dataset---------------------------------------------- #
        #jz default unifrom=True
        # # self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, uniform=None, class_choice=args.class_choice)
        # # TODO to refine here
        # class_choice = ['Airplane','Car','Chair','Table']
        # ratio = [500] * 4
        # self.data = ShapeNet_v0(root=args.dataset_path, npoints=args.point_num, uniform=None, class_choice=class_choice,ratio=ratio)
        # # TODO num workers to change back to 4
        # self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
        # print("Training Dataset : {} prepared.".format(len(self.data)))
        # # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
                     
        #jz parallel
        # self.G = nn.DataParallel(self.G)
        # self.D = nn.DataParallel(self.D)

        # self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        # self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))
        # #jz TODO check if think can be speed up via multi-GPU
        # self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
        # print("Network prepared.")
        # import pdb; pdb.set_trace()
    
    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):
        
        # loss_log = {'G_loss': [], 'D_loss': []}
        # loss_legend = list(loss_log.keys())

        # metric = {'FPD': []}
        # epoch_log = 0
        # if load_ckpt is not None:
        #     checkpoint = torch.load(load_ckpt, map_location=self.args.device)
        #     self.D.load_state_dict(checkpoint['D_state_dict'])
        #     self.G.load_state_dict(checkpoint['G_state_dict'])

        #     epoch_log = checkpoint['epoch']

        #     loss_log['G_loss'] = checkpoint['G_loss']
        #     loss_log['D_loss'] = checkpoint['D_loss']
        #     loss_legend = list(loss_log.keys())

        #     metric['FPD'] = checkpoint['FPD']
            
        #     print("Checkpoint loaded.")
        
        for epoch in range(epoch_log, self.args.epochs):
            # epoch_g_loss = []
            # epoch_d_loss = []
            # epoch_time = time.time()
            # for _iter, data in enumerate(self.dataLoader):
            #     # NOTE change args.batchsize into labels.shape[0], due to the last batch
                # Start Time
                # if _iter < 270:
                #     continue
                # # TODO remove
                # if _iter > 10:
                #     break
                
                # start_time = time.time()
                # point, labels = data
                # # TODO to work on treeGCN batch size issue.
                # if labels.shape[0] != self.args.batch_size:
                #     continue
                # point = point.to(self.args.device)
                # labels = labels.to(self.args.device)
                # labels_onehot = torch.FloatTensor(labels.shape[0], args.n_classes).to(self.args.device)
                # labels_onehot.zero_()
                # labels_onehot.scatter_(1, labels, 1)
                # labels_onehot.unsqueeze_(1)
                # -------------------- Discriminator -------------------- #
                tic = time.time()
                for d_iter in range(self.args.D_iter):
                    self.D.zero_grad()
                    # in tree-gan: normal distribution with mean 0, variance 1.
                    # in r-gan: mean 0 , sigma 0.2. link https://github.com/optas/latent_3d_points/blob/master/notebooks/train_raw_gan.ipynb
                    # in GCN GAN: sigma 0.2 https://github.com/diegovalsesia/GraphCNN-GAN/blob/master/gconv_up_aggr_code/main.py
                    # z = torch.randn(labels.shape[0], 1, 96).to(self.args.device)
                    # gen_labels = torch.from_numpy(np.random.randint(0, args.n_classes, labels.shape[0]).reshape(-1,1)).to(self.args.device)
                    # gen_labels_onehot = torch.FloatTensor(labels.shape[0], args.n_classes).to(self.args.device)
                    # gen_labels_onehot.zero_()
                    # gen_labels_onehot.scatter_(1, gen_labels, 1)
                    # gen_labels_onehot.unsqueeze_(1)

                    # NOTE: type may not be compatible
                    # # import pdb; pdb.set_trace()
                    # # print('iter and  update d',_iter,d_iter)
                    # tree = [z]
                    
                    # with torch.no_grad():
                    #     fake_point = self.G(tree,gen_labels_onehot)         
                        
                    
                    # D_realm = D_real.mean()

                    
                    # D_fakem = D_fake.mean()

                    # TODO try remove gp_loss and see how (tried, loss explode)
                    # gp_loss = self.GP(self.D, point.data, fake_point.data, conditional=True, yreal=labels_onehot)
                    
                    # d_loss = -D_realm + D_fakem
                    # d_loss_gp = d_loss + gp_loss
                    # d_loss_gp.backward()
                    # self.optimizerD.step()

                # loss_log['D_loss'].append(d_loss.item())   
                # epoch_d_loss.append(d_loss.item())          
                # toc = time.time()
                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()
                
                z = torch.randn(labels.shape[0], 1, 96).to(self.args.device)
                gen_labels = torch.from_numpy(np.random.randint(0, args.n_classes, labels.shape[0]).reshape(-1,1)).to(self.args.device)
                gen_labels_onehot = torch.FloatTensor(labels.shape[0], args.n_classes).to(self.args.device)
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
                # fake_pointclouds = torch.Tensor([])
                # jz, adjust for different batch size
                # TODO change back to 5000
                # test_batch_num = int(2000/self.args.batch_size)
                # for i in range(test_batch_num): # For 5000 samples
                    # print(i)
                    z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
                    gen_labels = torch.from_numpy(np.random.randint(0, args.n_classes, self.args.batch_size).reshape(-1,1)).to(self.args.device)
                    gen_labels_onehot = torch.FloatTensor(self.args.batch_size, args.n_classes).to(self.args.device)
                    gen_labels_onehot.zero_()
                    gen_labels_onehot.scatter_(1, gen_labels, 1)
                    gen_labels_onehot.unsqueeze_(1)
                    tree = [z]
                    with torch.no_grad():
                        sample = self.G(tree,gen_labels_onehot).cpu()
                    fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)

                fpd = calculate_fpd(fake_pointclouds, statistic_save_path=self.args.FPD_path, batch_size=100, dims=1808, device=self.args.device)
                metric['FPD'].append(fpd)
                print('-------------------------[{:4} Epoch] Frechet Pointcloud Distance <<< {:.2f} >>>'.format(epoch, fpd))

                class_name = args.class_choice if args.class_choice is not None else 'all'
                # TODO to enable save fake points
                # torch.save(fake_pointclouds, result_path+str(epoch)+'_'+class_name+'.pt')
                del fake_pointclouds

            # ---------------------- Save checkpoint --------------------- #
            class_name = args.class_choice if args.class_choice is not None else 'all'
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
        # self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        # Dataset arguments
        # self._parser.add_argument('--dataset_path', type=str, default='/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0', help='Dataset file path.')
        # self._parser.add_argument('--class_choice', type=str, default='Chair', help='Select one class to generate. [Airplane, Chair, ...] (default:all_class)')
        # #jz TODO batch size default 20
        # self._parser.add_argument('--batch_size', type=int, default=64, help='Integer value for batch size.')
        # self._parser.add_argument('--point_num', type=int, default=2048, help='Integer value for number of points.')

        # Training arguments
        # self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        #jz TODO epoch default 2000
        # self._parser.add_argument('--epochs', type=int, default=2, help='Integer value for epochs.')
        # self._parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
        #jz NOTE if do grid search, need to specify the ckpt_path and ckpt_save.
        # self._parser.add_argument('--ckpt_path', type=str, default='./model/checkpoints_cgan_v0/', help='Checkpoint path.')
        # self._parser.add_argument('--ckpt_save', type=str, default='tree_ckpt_', help='Checkpoint name to save.')
        # self._parser.add_argument('--ckpt_load', type=str, help='Checkpoint name to load. (default:None)')
        # self._parser.add_argument('--result_path', type=str, default='./model/generated_cgan_v0/', help='Generated results path.')
        # self._parser.add_argument('--result_save', type=str, default='tree_pc_', help='Generated results name to save.')
        # self._parser.add_argument('--visdom_port', type=int, default=8097, help='Visdom port number. (default:8097)')
        # self._parser.add_argument('--visdom_color', type=int, default=4, help='Number of colors for visdom pointcloud visualization. (default:4)')

        # Network arguments
        # self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        # self._parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
        # self._parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
        # self._parser.add_argument('--DEGREE', type=int, default=[1,  2,   2,   2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
        # self._parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
        #jz default is default=[3,  64,  128, 256, 512, 1024]
        # for D in r-GAN
        # self._parser.add_argument('--D_FEAT', type=int, default=[3, 64,  128, 256, 256, 512], nargs='+', help='Features for discriminator.')
        # for D in source code
        # self._parser.add_argument('--D_FEAT', type=int, default=[3,  64,  128, 256, 512, 1024], nargs='+', help='Features for discriminator.')
        # Evaluation arguments
        # self._parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')
        # self._parser.add_argument('--n_classes',type=int, default=16)
        self._parser.add_argument('--version', type=int, default=0)

    def parser(self):
        return self._parser


# if __name__ == '__main__':
#     opt = Arguments().parser().parse_args()
#     opt.device = torch.device('cuda:'+str(opt.gpu) if torch.cuda.is_available() else 'cpu')
#     torch.cuda.set_device(opt.device)

#     SAVE_CHECKPOINT = opt.ckpt_path + opt.ckpt_save if opt.ckpt_save is not None else None
#     LOAD_CHECKPOINT = opt.ckpt_load if opt.ckpt_load is not None else None #NOTE: changed for more flexible loading
#     RESULT_PATH = opt.result_path + opt.result_save

#     model = ConditionalTreeGAN(opt)
#     model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT, result_path=RESULT_PATH)