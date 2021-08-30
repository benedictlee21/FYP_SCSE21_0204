"""
This file evaluate generated points from a GAN compare with 

It first generate point sets from a given GAN

It should be saving generated points

It should also be 

"""
import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_benchmark import BenchmarkDataset
from datasets import ShapeNet_v0
from model.gan_network import Generator, Discriminator
from train_cgan import ConditionalGenerator_v0
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd, calculate_activation_statistics

from metrics import *
from loss import *

from evaluation.pointnet import PointNetCls
from math import ceil
# from arguments import Arguments
import argparse
import time
import visdom
import numpy as np
import time
import os.path as osp
import os
# import Namespace
import copy
from utils.common_utils import *

def count_shapenet_v0():
    root_dir = '/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0'
    catfile = './data/synsetoffset2category.txt'
    class2dir_dict = {}
    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            class2dir_dict[ls[0]] = ls[1]
    class2id_dict = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 
    'Earphone': 5, 'Guitar': 6,  'Knife': 7, 'Lamp': 8, 'Laptop': 9, 
    'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}   
    count_list = [0] * 16
    class2cnt_dict = {}
    print(class2dir_dict)
    print(class2id_dict)
    for class_name, id in class2id_dict.items():
        dir_point = os.path.join(root_dir, class2dir_dict[class_name], 'points')
        fns = os.listdir(dir_point)
        count_list[id] = len(fns)
        class2cnt_dict[class_name] = len(fns)
    print(class2cnt_dict)
    print(np.sum(count_list))
    return count_list


def save_pcs_to_txt(save_dir, fake_pcs, labels=None):
    sample_size = fake_pcs.shape[0]
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    for i in range(sample_size):
        if labels.shape[0] == 0:
            np.savetxt(osp.join(save_dir,str(i)+'.txt'), fake_pcs[i], fmt = "%f;%f;%f")  
        else:
            class_id = labels[i].detach().cpu().numpy()
            np.savetxt(osp.join(save_dir,str(i)+'_'+str(class_id)+'.txt'), fake_pcs[i], fmt = "%f;%f;%f") 

def generate_pcs(model_cuda, n_pcs=5000, batch_size=64, n_classes=1,ratio=None, conditional=False,device=None):
    # import pdb; pdb.set_trace()
    fake_pcs = torch.Tensor([])
    all_gen_labels = torch.Tensor([])
    n_pcs = int(ceil(n_pcs/batch_size) * batch_size)
    n_batches = ceil(n_pcs/batch_size)
    if not conditional:
    # if n_classes == 1 or ratio is None:
        for i in range(n_batches):
            z = torch.randn(batch_size, 1, 96).to(device)
            # tree = [z]
            with torch.no_grad():
                sample = model_cuda(z).cpu()
            fake_pcs = torch.cat((fake_pcs, sample), dim=0)
    elif conditional and ratio is False:
        # TODO conditional need to refine here
        for i in range(n_batches):
            # import pdb; pdb.set_trace()
            z = torch.randn(batch_size, 1, 96).to(device)
            gen_labels = torch.from_numpy(np.random.randint(0, opt.n_classes, batch_size).reshape(-1,1)).to(device)
            all_gen_labels = torch.cat((all_gen_labels,gen_labels.cpu()),0)
            gen_labels_onehot = torch.FloatTensor(batch_size, opt.n_classes).to(device)
            gen_labels_onehot.zero_()
            gen_labels_onehot.scatter_(1, gen_labels, 1)
            gen_labels_onehot.unsqueeze_(1)
            # tree = [z]
            with torch.no_grad():
                sample = model_cuda(z,gen_labels_onehot).cpu()
            fake_pcs = torch.cat((fake_pcs, sample), dim=0)
    else:
        # n_pcs = 300
        # NOTE: due to non-whole batch resulting issue in some models, round up n_pcs
        
        # got ratio, assume which is a list ofßß counts from the training data
        # print (n_pcs)
        # NOTE, here shoulb change if n_classes changed
        # ratio = count_shapenet_v0()
        ratio = [500] * 4
        # print (ratio)
        ratio_nm = np.array(ratio)/np.sum(ratio)
        # print (ratio_nm)
        ratio_cnt = ratio_nm * n_pcs
        # just check all chair scenario 
        # ratio_cnt = [0] * 4 + [5056] + [0] * 11
        # print (ratio_cnt)
        all_gen_labels = torch.zeros(n_pcs).type(torch.LongTensor).reshape(-1,1)
        # NOTE due to some r is not int, there might be last a few all_gen_labels value remain at 0.
        # TODO to shuffle the tensor
        pointer = 0
        for i, r in enumerate(ratio_cnt):
            all_gen_labels[pointer:(pointer+int(r))] = int(i)
            pointer+=int(r)
        # print(all_gen_labels)
        for i in range(n_batches):
            # import pdb; pdb.set_trace()
            z = torch.randn(batch_size, 1, 96).to(device)
            gen_labels = all_gen_labels[i*batch_size:(i+1)*batch_size].reshape(-1,1).to(device)
            # print(gen_labels.dtype)
            gen_labels_onehot = torch.FloatTensor(batch_size, opt.n_classes).to(device)
            gen_labels_onehot.zero_()
            gen_labels_onehot.scatter_(1, gen_labels, 1)
            gen_labels_onehot.unsqueeze_(1)
            # tree = [z]
            with torch.no_grad():
                sample = model_cuda(z,gen_labels_onehot).cpu()
            fake_pcs = torch.cat((fake_pcs, sample), dim=0)

    return fake_pcs, all_gen_labels

def create_fpd_stats(pcs, pathname_save, device):
    # pcs = pcs.transpose(1,2)
    PointNet_pretrained_path = './evaluation/cls_model_39.pth'

    model = PointNetCls(k=16).to(device)
    model.load_state_dict(torch.load(PointNet_pretrained_path))
    mu, sigma = calculate_activation_statistics(pcs, model, device=device)
    print (mu.shape, sigma.shape)
    np.savez(pathname_save,m=mu,s=sigma)
    print('saved into', pathname_save)
    # f = np.load(pathname_save)
    # m2, s2 = f['m'][:], f['s'][:]
    # f.close()
    # print('loading, m2, s2 shape',m2.shape, s2.shape)

def script_create_fpd_stats(opt):
    pathname_save = './evaluation/pre_statistics_4x500.npz'
    class_choice = ['Airplane','Car','Chair','Table']
    ratio = [500] * 4
    dataset = ShapeNet_v0(root=opt.dataset_path, npoints=opt.point_num, uniform=None, class_choice=class_choice,ratio=ratio)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    ref_pcs = torch.Tensor([])
    for _iter, data in enumerate(dataLoader):
        point, labels = data
        ref_pcs = torch.cat((ref_pcs, point),0)
    print ('shape of ref_pcs',ref_pcs.shape)
    create_fpd_stats(ref_pcs,pathname_save, opt.device)

@timeit
def checkpoint_eval(G_net, device, n_samples=5000, batch_size=100,conditional=False, FPD_path=None):
    """
    an abstraction used during training
    """
    G_net.eval()
    fake_pcs, labels = generate_pcs(G_net, n_pcs=n_samples, batch_size=batch_size, conditional=conditional, device=device, ratio = None)
    fpd = calculate_fpd(fake_pcs, statistic_save_path=FPD_path, batch_size=100, dims=1808, device=device)
    # print(fpd)
    print('----------------------------------------- Frechet Pointcloud Distance <<< {:.2f} >>>'.format(fpd))

def test(opt, modes='FJMCS', verbose=True):
    '''
    NOTE: model is of a certain class now
    args needed: 
        n_classes, pcs to generate, ratio of each class, class to id dict???
        model pth, , points to save, save pth, npz for the class, 
    '''
    # print(' in FPD')
    # print('')
    if not opt.conditional:
        G_net = Generator(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support,version=opt.version,args=opt).to(device)
    else:
        G_net = ConditionalGenerator_v0(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support, n_classes=opt.n_classes,version=opt.version).to(opt.device)
    print(G_net)
    # print(opt.model_pathname, opt.version)
    checkpoint = torch.load(opt.model_pathname, map_location=device)
    try:
        G_net.load_state_dict(checkpoint['G_state_dict'])
    except:
        G_net = nn.DataParallel(G_net)
        G_net.load_state_dict(checkpoint['G_state_dict'])
    G_net.eval()
    # compute ratio 
    # if not conditional, labels are dummy
    fake_pcs, labels = generate_pcs(G_net, n_pcs = opt.n_samples, batch_size = opt.batch_size, conditional=opt.conditional, device=opt.device, ratio = opt.conditional_ratio)
    # print('fake_pcs shape,',fake_pcs.shape)
    ans = {}
    if 'S' in modes:
        # import pdb; pdb.set_trace()
        save_pcs_to_txt(opt.save_sample_path, fake_pcs, labels=labels)
    if 'F' in modes:
        # TODO check all-chair only scenario
        # opt.FPD_path = './evaluation/pre_statistics_chair.npz'
        fpd = calculate_fpd(fake_pcs, statistic_save_path=opt.FPD_path, batch_size=100, dims=1808, device=opt.device)
        # print('Frechet Pointcloud Distance <<< {:.4f} >>>'.format(fpd))
        ans['fpd'] = fpd
    if 'J' in modes or 'M' in modes or 'C' in modes:
        if opt.mmd_cov_loss == 'cd':
            use_EMD = False
        else:
            use_EMD = True
        ans['loss'] = opt.mmd_cov_loss
        batch_size = opt.batch_size 
        normalize = True
        # get point clouds for a particular data
        gt_dataset = BenchmarkDataset(root=opt.dataset_path, npoints=2048, uniform=None, class_choice=opt.class_choice)
        dataLoader = torch.utils.data.DataLoader(gt_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=10)
        # gt_data_list = []
        gt_data = torch.Tensor([])
        for _iter, data in enumerate(dataLoader):
            point, _  = data
            # gt_data_list.append(point)
            gt_data = torch.cat((gt_data,point),0)
            if gt_data.shape[0] >= opt.mmd_cov_jsd_ref_num:
                break
        # TODO not suffled yet
        ref_pcs = gt_data[:opt.mmd_cov_jsd_ref_num].detach().cpu().numpy()
        sample_pcs = fake_pcs[:opt.mmd_cov_jsd_sample_num].detach().cpu().numpy()
        if 'J' in modes:
            tic = time.time()
            jsd = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)
            ans['jsd'] = jsd
            toc = time.time()
            if verbose:
                print('time spent in JSD',int(toc-tic))
            # # jsd1 = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)
            # # jsd2 = jsd_between_point_cloud_sets(ref_pcs[:3000], ref_pcs, resolution=28)
            # # jsd3 = jsd_between_point_cloud_sets(sample_pcs[:2000], ref_pcs[-1000:], resolution=28)
            # # print ('jsd1, 2, 3', jsd1, jsd2, jsd3)
            # # # jsd1, 2, 3 0.11505738867010251 0.0002315239257608681 0.1166695618494149  
        if 'M' in modes:
            tic = time.time()
            if opt.mmd_cov_loss_batch:
                mmd, matched_dists = MMD_batch(sample_pcs,ref_pcs,batch_size, normalize=normalize, use_EMD=use_EMD,device=opt.device)
            else:
                mmd, matched_dists = minimum_mathing_distance(sample_pcs,ref_pcs,batch_size, normalize=normalize, use_EMD=use_EMD,device=opt.device)
            ans['mmd'] = jsd
            toc = time.time()
            if verbose:
                print('time spent in MMD',int(toc-tic))
        if 'C' in modes:
            tic = time.time()
            cov, matched_loc, matched_dist = coverage(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=use_EMD,device=opt.device)
            ans['cov'] = cov   
            toc = time.time()
            if verbose:
                print('time spent in Cov',int(toc-tic))   
    return ans


if __name__ == '__main__':
    # def visualize_pcd_to_png()


    parser = argparse.ArgumentParser()
    ### paths
    parser.add_argument('--dataset_path', type=str, default='/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0', help='Dataset file path.')
    parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')
    # parser.add_argument('--save_model_path')
    parser.add_argument('--load_model_path',required=True,help='path of the GAN to be evaled')
    parser.add_argument('--save_sample_path',required=True,help='dir to save generated point clouds')   
    parser.add_argument('--model_pathname', type=str,default='None',help='./model/checkpoints18/tree_ckpt_1660_Chair.pt')
    parser.add_argument('--epoch_load', type=int,required=True,help='./model/checkpoints18/tree_ckpt_1660_Chair.pt')
    ### model ralated
    parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
    parser.add_argument('--DEGREE', type=int, default=[1,  2,   2,   2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
    parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
    parser.add_argument('--conditional', default=False, type=lambda x: (str(x).lower() == 'true'))  
    parser.add_argument('--n_classes',type=int, default=16) # TODO
    parser.add_argument('--conditional_ratio',default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--version', type=int, default=0)
    ### general
    parser.add_argument('--n_points', type=int, default=2048, help='old: point_num')
    parser.add_argument('--class_choice', type=str, default='Chair', help='Select one class to generate. [Airplane, Chair, ...] (default:all_class)') # TODO
    parser.add_argument('--batch_size', type=int, default=64, help='Integer value for batch size.') 
    ### test related
    parser.add_argument('--mmd_cov_loss',type=str,default='cd')
    parser.add_argument('--mmd_cov_loss_batch',default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--mmd_cov_jsd_ref_num',type=int,default=100) #TODO 
    parser.add_argument('--mmd_cov_jsd_sample_num',type=int,default=100) #TODO
    parser.add_argument('--test_modes',type=str,default='FJMCS') #TODO
    parser.add_argument('--n_samples',type=int, default=5000, help='number for points to be generated by the G')
    parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--degrees_opt', type=str, default='default', help='Upsample degrees for generator.')



    opt = parser.parse_args()
    device = torch.device('cuda')
    opt.device = device
    print(type(opt))
    print(opt)
    print(opt.test_modes)
    # print(opt.conditional,type(opt.conditional))
    # print(opt.conditional_ratio,type(opt.conditional_ratio))
    tic  = time.time()

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

    ################################# above is common section


    ########################Section test()
    # epoch_checkpoints = [965]
    epoch_checkpoints = [opt.epoch_load]
    # epoch_checkpoints = range(2000, 0, -100)
    for epoch in epoch_checkpoints:
        if epoch == 2000:
            epoch = 1995
        opt_deep_copy = copy.deepcopy(opt)
        tic = time.time()
        opt_deep_copy.model_pathname = opt_deep_copy.load_model_path + '/tree_ckpt_'+str(epoch)+'_'+str(opt.class_choice)+'.pt'
        print('model_pathname',opt_deep_copy.model_pathname)
        results = test(opt_deep_copy,modes=opt.test_modes)
        toc = time.time()
        # print ('--------------------time spent:',int(toc-tic),'|| epoch:',epoch,'|| FPD: <<< {:.2f} >>>'.format(fpd_value))
        print('results',results)


    ########################Section check uniform loss
    # plane_dir = '/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points'

    # plane_name = '6c8275f09052bf66ca8607f540cc62ba.pts'

    # # # # cp /mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points/6c8275f09052bf66ca8607f540cc62ba.pts ~

    # plane_pathname = osp.join(plane_dir,plane_name)

    # point_set = np.loadtxt(plane_pathname).astype(np.float32)
    # print('pc shape',point_set.shape)
    # pc_t = torch.from_numpy(point_set)
    # pcs = torch.stack([pc_t]*64)
    # print('pcs shape',pcs.shape)
    # seeds = farthest_point_sample(pcs,10) # returned index, not coordinates.
    # print('seeds shape',seeds.shape)
    # print(seeds[0])
    # seed_points = point_set[seeds[0]]
    # print(seed_points)
    # patches = extract_knn_patch(seed_points,point_set,50)
    # print(patches[0].shape)
    # print('===================')
    # # print(patches)
    # radius = 0.005
    # nsample = 500
    # new_xyz = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)])
    # print('new_xyz shape',new_xyz.shape)

    # group_idx = query_ball_point(radius, nsample, pcs, new_xyz)


    ########################Section
    # check classification
    ###
    # PointNet_pretrained_path = './evaluation/cls_model_39.pth'
    # model = PointNetCls(k=16)
    # model.load_state_dict(torch.load(PointNet_pretrained_path))
    # model.to(device)
    # # fake_pointclouds = fake_pointclouds.transpose(1,2)
    # # soft, trans, actv = model(fake_pointclouds.to(device))
    # # import pdb; pdb.set_trace()

    # dataset = BenchmarkDataset(root=opt.dataset_path, npoints=2048, uniform=None, class_choice=opt.class_choice)
    # dataLoader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # for _iter, data in enumerate(dataLoader):
    #     point, _ = data
    #     point = point.transpose(1,2).to(opt.device)
    #     soft, trans, actv = model(point)
    #     import pdb; pdb.set_trace()

    ########################Section check kNN distance
    # dataset = BenchmarkDataset(root=opt.dataset_path, npoints=2048, uniform=None, class_choice=opt.class_choice)
    # dataLoader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=10)
    # # pcs_cpu = torch.Tensor([])
    # n_seeds = 100
    # k = 1
    # for _iter, data in enumerate(dataLoader):
    #     pcs, _ = data
    #     # pcs_cpu = torch.cat((pcs_cpu,point),0)
    #     seeds = farthest_point_sample(pcs,n_seeds)
    #     seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)])
    #     pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
    #     seeds_new = seeds_value.unsqueeze(1).repeat(1,2048,1,1)
    #     # print(pcs_new.shape,seeds_new.shape)
    #     dist = pcs_new.add(-seeds_new)
    #     dist_value = torch.norm(dist,dim=3)
    #     # print(dist.shape)
    #     # print(dist_value.shape)
    #     toc = time.time()
    #     dist_new = dist_value.transpose(1,2)
    #     tac = time.time()
    #     # print(dist_new.shape)
    #     top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False)
    #     # print('return ans',idx.shape, top_dist.shape)
    #     overall_mean = top_dist[:,:,1:].mean()
    #     print(overall_mean)
    #     print(top_dist[0,:,1])
    #     import pdb; pdb.set_trace()
        



    ########################Section check variance
    input_data = False
    generated_data = False
    r_list = [0.005, 0.01, 0.02, 0.05, 0.08, 0.1]
    # n_seeds_list = [200, 100, 50, 20, 12, 10]
    # n_seeds_list = [400, 200, 100, 40, 25]
    # n_seeds_list = [50, 50, 50, 50, 50, 50]
    n_seeds_list = [200] * 6
    var_version = 2
    if input_data:
        for radius, n_seeds in zip(r_list, n_seeds_list):
            dataset = BenchmarkDataset(root=opt.dataset_path, npoints=2048, uniform=None, class_choice=opt.class_choice)
            dataLoader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=10)
            pcs_cpu = torch.Tensor([])
            for _iter, data in enumerate(dataLoader):
                point, _ = data
                pcs_cpu = torch.cat((pcs_cpu,point),0)
            pcs = pcs_cpu.to(opt.device)
            var = cal_patch_density_variance(pcs, radius=radius, n_seeds=n_seeds, batch_size=100, device=opt.device,version=var_version)        
            print('-----------------radius, seeds:',radius, n_seeds, 'var: {:.10f}'.format(var.cpu().numpy()))
    print('below generated data')
    if generated_data:
        # model_dir_list = ['./model/uni_loss_v'+str(i) for i in range(9)]
        # epoch_list = [1975, 1995, 1660, 1475, 1480, 1460, 1655, 1895, 1290]
        ## baseline
        model_dir_list = ['./model/checkpoints18']
        epoch_list = [1660]
        # model_dir_list = ['./model/uni_loss_v'+str(i) for i in range(9,17)]
        # epoch_list = [440, 1295, 1095, 1185, 1015, 1035, 810, 935]
        for radius, n_seeds in zip(r_list, n_seeds_list):
            for model_dir, epoch in zip(model_dir_list,epoch_list):
                # radius = 0.01
                # n_seeds = 30
                opt.model_pathname = osp.join(model_dir,'tree_ckpt_'+str(epoch)+'_Chair.pt')
                if not opt.conditional:
                    G_net = Generator(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support,version=opt.version,args=opt).to(device)
                else:
                    G_net = ConditionalGenerator_v0(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support, n_classes=opt.n_classes,version=opt.version).to(opt.device)
                # print(G_net)
                # print(opt.model_pathname, opt.version)
                checkpoint = torch.load(opt.model_pathname, map_location=device)
                G_net.load_state_dict(checkpoint['G_state_dict'])
                G_net.eval()
                # compute ratio 
                # if not conditional, labels are dummy
                pcs, labels = generate_pcs(G_net, n_pcs = opt.n_samples, batch_size = opt.batch_size, conditional=opt.conditional, device=opt.device, ratio = opt.conditional_ratio)
                # print('fake_pcs shape,',fake_pcs.shape)
                var = cal_patch_density_variance(pcs, radius=radius, n_seeds=n_seeds, batch_size=100, device=opt.device,version=var_version)        
                print(model_dir[8:], epoch,' - r:',radius,'seeds:',n_seeds, '== var: {:.10f}'.format(var.cpu().numpy()))