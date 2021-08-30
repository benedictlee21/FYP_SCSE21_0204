import numpy as np
# from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Function
try:
    import expansion_penalty
except:
    print('no expansion penality module')

### below is chamfer distance from: https://github.com/lynetcha/completion3d 
# the chamfer source file is in pcd_lib/chamfer
import math
import sys
from numbers import Number
from collections import Set, Mapping, deque
try:
    import chamfer
except:
    print('no chamfer module')
### below is EMD from https://github.com/daerduoCarey/PyTorchEMD 
try:
    import emd_cuda
except:
    print('no EMD module')
### 
def square_distance(src, dst):
    # link: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    """
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2     
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) # xm*xm + ym*ym + zm*zm
    return dist
def farthest_point_sample(xyz, npoint):
    # link: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    # form: torch tensor
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
    	# Update the i-th farthest point
        centroids[:, i] = farthest
        # Take the xyz coordinate of the farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # Calculate the Euclidean distance from all points in the point set to this farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distances to record the minimum distance of each point in the sample from all existing sample points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, xyz, new_xyz, nsample=500, density_only=True):
    # link: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93    
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists: [B, S, N] Record the Euclidean distance between the center point and all points
    sqrdists = square_distance(new_xyz, xyz) # shape (B, S, N)
    # import pdb; pdb.set_trace()
    # Find all distances greater than radius^2, its group_idx is directly set to N; the rest retain the original value
    
    if not density_only:
        group_idx[sqrdists > radius ** 2] = N
        # Do ascending order, the front is greater than radius^2 are N, will be the maximum, so will take the first nsample points directly in the remaining points
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        # Considering that there may be points in the previous nsample points that are assigned N (ie, less than nsample points in the spherical area), this point needs to be discarded, and the first point can be used instead.
        # group_first: [B, S, k], actually copy the value of the first point in group_idx to the dimension of [B, S, K], which is convenient for subsequent replacement.
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        # Find the point where group_idx is equal to N
        mask = group_idx == N
        # Replace the value of these points with the value of the first point
        group_idx[mask] = group_first[mask]
        return group_idx
    else:
        raw_mat = torch.zeros(B,S,N)
        density_mat = torch.zeros(B,S)
        raw_mat[sqrdists <= radius ** 2] = 1
        density_mat = torch.sum(raw_mat,2)
        # print(torch.max(sqrdists))
        return density_mat

# def extract_knn_patch(queries, pc, k):
#     """
#     queries [M, C]
#     pc [P, C]
#     """
#     knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
#     knn_search.fit(pc)
#     knn_idx = knn_search.kneighbors(queries, return_distance=False)
#     k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
#     return k_patches

def get_pairwise_distance(batch_features):
    # link: https://github.com/liruihui/PU-GAN/blob/master/Common/pc_util.py
    """Compute pairwise distance of a point cloud.
    Args:
      batch_features: numpy (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    og_batch_size = len(batch_features.shape)

    if og_batch_size == 2: #just two dimension
        batch_features = np.expand_dims(batch_features, axis=0)


    batch_features_transpose = np.transpose(batch_features, (0, 2, 1))

    #batch_features_inner = batch_features@batch_features_transpose
    batch_features_inner = np.matmul(batch_features,batch_features_transpose)

    #print(np.max(batch_features_inner), np.min(batch_features_inner))


    batch_features_inner = -2 * batch_features_inner
    batch_features_square = np.sum(np.square(batch_features), axis=-1, keepdims=True)


    batch_features_square_tranpose = np.transpose(batch_features_square, (0, 2, 1))

    return batch_features_square + batch_features_inner + batch_features_square_tranpose

def py_uniform_loss(points,idx,pts_cn,radius):
    '''
    from PU GAN: py version, link: https://github.com/liruihui/PU-GAN/blob/master/Common/loss_utils.py
    points: a batch of pcd
    idx??
    pts ??
    radius 1
    ----------------
    nsample : n_hat
    how to get n_hat
    pts_cn
    idx : should be returned value. which (B,M,nsample???), which is variable

    '''
    #print(type(idx))
    B,N,C = points.shape
    _,npoint,nsample = idx.shape
    uniform_vals = []
    for i in range(B):
        point = points[i]
        for j in range(npoint):
            number = pts_cn[i,j]
            coverage = np.square(number - nsample) / nsample
            if number<5:
                uniform_vals.append(coverage)
                continue
            _idx = idx[i, j, :number]
            disk_point = point[_idx]
            if disk_point.shape[0]<0:
                pair_dis = get_pairwise_distance(disk_point)#(batch_size, num_points, num_points)
                nan_valid = np.where(pair_dis<1e-7)
                pair_dis[nan_valid]=0
                pair_dis = np.squeeze(pair_dis, axis=0)
                pair_dis = np.sort(pair_dis, axis=1)
                shortest_dis = np.sqrt(pair_dis[:, 1])
            else:
                shortest_dis = pc_util.get_knn_dis(disk_point,disk_point,2)
                shortest_dis = shortest_dis[:,1]
            disk_area = math.pi * (radius ** 2) / disk_point.shape[0]
            #expect_d = math.sqrt(disk_area)
            expect_d = np.sqrt(2 * disk_area / 1.732)  # using hexagon
            dis = np.square(shortest_dis - expect_d) / expect_d
            uniform_val = coverage * np.mean(dis)

            uniform_vals.append(uniform_val)

    uniform_dis = np.array(uniform_vals).astype(np.float32)

    uniform_dis = np.mean(uniform_dis)
    return uniform_dis

### uniform loss from PU-GAN: link https://github.com/liruihui/PU-GAN/blob/master/Common/loss_utils.py
# def get_uniform_loss(pcd, percentages=[0.004,0.006,0.008,0.010,0.012], radius=1.0):
#     B, N, C = pcd.shape
#     npoint = int(N * 0.05)
#     loss=[]
#     for p in percentages:
#         # TODO to figure out the below equations ???
#         nsample = int(N*p)
#         r = math.sqrt(p*radius) # ??? not sure if it is corect or not
#         disk_area = math.pi *(radius ** 2) * p/nsample
#         #print(npoint,nsample)
#         new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
#         idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)#(batch_size, npoint, nsample)

class kNNRepulsionLoss(nn.Module):
    def __init__(self, k=10, n_seeds=20, h=0.01):
        super(kNNRepulsionLoss,self).__init__()
        self.k = k
        self.n_seeds = n_seeds
        self.h = h
    def forward(self, pcs):
        tic = time.time()
        n_seeds = self.n_seeds
        k = self.k
        seeds = farthest_point_sample(pcs,n_seeds) # which gives index
        temp = time.time()
        # print('time in fps',temp-tic) # 0.05 for 100, 0.05 for 20 also, bottlenet!!
        seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) # grad
        # print('shape of pcs and seeds',pcs.shape,seeds.shape,seeds_value.shape)

        # dist = pcs - seeds
        # dist = pcs.add(-seeds)
        pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
        seeds_new = seeds_value.unsqueeze(1).repeat(1,2048,1,1)
        # print(pcs_new.shape,seeds_new.shape)
        dist = pcs_new.add(-seeds_new)
        dist_value = torch.norm(dist,dim=3)
        # print(dist.shape)
        # print(dist_value.shape)
        toc = time.time()
        dist_new = dist_value.transpose(1,2)
        tac = time.time()
        # print(dist_new.shape)
        top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False)
        top_dist_net = top_dist[:,:,1:]
        weights = torch.exp(-torch.pow(top_dist_net,2)*(1/(self.h**2)))
        repulsion = torch.mul(-top_dist_net,weights)
        # import pdb; pdb.set_trace()
        return repulsion.sum(2).sum(1).mean()



class kNNLoss(nn.Module):
    def __init__(self, k=10, n_seeds=20):
        super(kNNLoss,self).__init__()
        self.k = k
        self.n_seeds = n_seeds
    def forward(self, pcs):
        tic = time.time()
        n_seeds = self.n_seeds
        k = self.k
        seeds = farthest_point_sample(pcs,n_seeds) # which gives index
        temp = time.time()
        # print('time in fps',temp-tic) # 0.05 for 100, 0.05 for 20 also, bottlenet!!
        seeds_value = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) # grad
        # print('shape of pcs and seeds',pcs.shape,seeds.shape,seeds_value.shape)

        # dist = pcs - seeds
        # dist = pcs.add(-seeds)
        pcs_new = pcs.unsqueeze(2).repeat(1,1,n_seeds,1)
        seeds_new = seeds_value.unsqueeze(1).repeat(1,2048,1,1)
        # print(pcs_new.shape,seeds_new.shape)
        dist = pcs_new.add(-seeds_new)
        dist_value = torch.norm(dist,dim=3)
        # print(dist.shape)
        # print(dist_value.shape)
        toc = time.time()
        dist_new = dist_value.transpose(1,2)
        tac = time.time()
        # print(dist_new.shape)
        top_dist, idx = torch.topk(dist_new, k+1, dim=2, largest=False)
        # print('return ans',idx.shape, top_dist.shape)
        overall_mean = top_dist[:,:,1:].mean()
        top_dist = top_dist/overall_mean
        var = torch.var(top_dist.mean(dim=2)).mean()
        tc = time.time()
        # print('times','total:',tc-tic,' 1:',toc-tic, '   2:',tac-toc,'   3:',tc-tac)
        # import pdb; pdb.set_trace()
        return var

class PatchRepulsionLoss(nn.Module):
    def __init__(self, n_patches=32, n_sigma=1):
        super(PatchRepulsionLoss, self).__init__()
        self.n_patches = n_patches
        self.n_per_patch = int(2048/n_patches)
        self.n_sigma = n_sigma
    def forward(self, pcs):
        means = [torch.mean(pcs[:,i*self.n_per_patch:self.n_per_patch*(i+1),:],1) for i in range(self.n_patches)]
        # vars = [torch.var(pcs[:,i*self.n_per_patch:self.n_per_patch*(i+1),:],1) for i in range(self.n_patches)]
        stds = [pcs[:,i*self.n_per_patch:self.n_per_patch*(i+1),:].std(dim=1) for i in range(self.n_patches)]
        # dists = []
        # vars_sum = []
        repulsion_list = []
        # TODO? below for loop can speed up??
        for i in range(self.n_patches):
            for j in range(i,self.n_patches):
                abs_dim_delta = torch.abs(means[i] - means[j])
                repulsion = nn.functional.relu((stds[i]+stds[j])*self.n_sigma - abs_dim_delta)
                repulsion_list.append(torch.norm(repulsion,dim=1))
        return torch.stack(repulsion_list).mean()        

class PatchDensityVariance(nn.Module):
    def __init__(self, radius=0.05, n_seeds=40):
        super(PatchDensityVariance, self).__init__()
        self.radius = radius
        self.n_seeds = n_seeds
    
    def forward(self, pcs):
        # TODO make it in the cuda??
        seeds = farthest_point_sample(pcs,self.n_seeds)
        new_xyz = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)]) # grad
        # density_mat = query_ball_point(self.radius, pcs, new_xyz) # false
        # den_var = torch.var(density_mat,1).mean() # false
        ### replace above with grad: sqrdists.requires_grad
        B, N, C = pcs.shape
        S = self.n_seeds
        # sqrdists = square_distance(new_xyz, pcs)
        
        sqrdists = -2 * torch.matmul(new_xyz, pcs.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
        sqrdists += torch.sum(new_xyz ** 2, -1).view(B, S, 1) # xn*xn + yn*yn + zn*zn
        sqrdists += torch.sum(pcs ** 2, -1).view(B, 1, N) # xm*xm + ym*ym + zm*zm
        
        

        ### verification only
        # raw_mat = torch.zeros(B,S,N) 
        #,requires_grad=True) a leaf Variable that requires grad has been used in an in-place operation.
        # density_mat = torch.zeros(B,S)
        # raw_mat[sqrdists <= self.radius ** 2] = 1
        # density_mat = torch.sum(raw_mat,2)
        # relu_dists = nn.functional.relu(sqrdists-self.radius ** 2)
        
        # softmax_op = torch.nn.Softmax()
        # softmax_dists =  softmax_op(relu_dists*100)
        # raw_mat = torch.ones(B,S,N)* softmax_dists
        # 
        # raw_mat = relu_dists[relu_dists>0]

        relu_dists = nn.functional.relu(self.radius ** 2-sqrdists)
        den_mat = torch.sum(relu_dists,2)
        den_var = torch.var(den_mat,1).mean()
        
        # import pdb; pdb.set_trace()
        return den_var

# GPU tensors only
class expansionPenaltyFunction(Function):
    @staticmethod
    def forward(ctx, xyz, primitive_size, alpha):
        # 512 change to primitive_size TODO
        assert(primitive_size <= 512)
        batchsize, n, _ = xyz.size()
        assert(n % primitive_size == 0)
        xyz = xyz.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        neighbor = torch.zeros(batchsize, n * 512,  device='cuda', dtype=torch.int32).contiguous()
        cost = torch.zeros(batchsize, n * 512, device='cuda').contiguous()
        mean_mst_length = torch.zeros(batchsize, device='cuda').contiguous()
        expansion_penalty.forward(xyz, primitive_size, assignment, dist, alpha, neighbor, cost, mean_mst_length)
        ctx.save_for_backward(xyz, assignment)
        return dist, assignment, mean_mst_length / (n / primitive_size)

    @staticmethod
    def backward(ctx, grad_dist, grad_idx, grad_mml):
        xyz, assignment = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()
        grad_xyz = torch.zeros(xyz.size(), device='cuda').contiguous()
        expansion_penalty.backward(xyz, grad_xyz, grad_dist, assignment)
        return grad_xyz, None, None

class expansionPenaltyModule(nn.Module):
    def __init__(self):
        super(expansionPenaltyModule, self).__init__()

    def forward(self, input, primitive_size, alpha):
        return expansionPenaltyFunction.apply(input, primitive_size, alpha)

class chamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.cuda()
        dist2 = dist2.cuda()
        idx1 = idx1.cuda()
        idx2 = idx2.cuda()

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.cuda()
        gradxyz2 = gradxyz2.cuda()
        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2

class chamferDist(nn.Module):
    def __init__(self):
        super(chamferDist, self).__init__()

    def forward(self, input1, input2):
        return chamferFunction.apply(input1, input2)

class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2

class EMD(nn.Module):
    def __init__(self):
        super(EMD, self).__init__()

    def forward(self, input1, input2):
        return EarthMoverDistanceFunction.apply(input1, input2)

# def test():
#     import time
#     x1 = torch.rand(20, 8192, 3).cuda()
#     x2 = torch.rand(20, 8192, 3).cuda()
#     cd = chamferDist()
#     start_time = time.perf_counter()
#     dist1, dist2  = cd(x1, x2)
#     print("Input_size: ", x1.shape)
#     print("Runtime: %lfs" % (time.perf_counter() - start_time))
#     print('shape of dist1 and dist2',dist1.shape,dist2.shape)
#     dist = (dist1+dist2).mean(dim=1)
#     print (dist.shape)

# test()

### from dgp NOTE
class PerceptLoss(object):

    def __init__(self):
        pass

    def __call__(self, LossNet, fake_img, real_img):
        fake_img_t = fake_img.transpose(1,2).repeat(32,1,1)
        real_img_t = real_img.transpose(1,2).repeat(32,1,1)
        with torch.no_grad():
            real_feature = LossNet(real_img_t.detach())
        fake_feature = LossNet(fake_img_t)
        perceptual_penalty = F.mse_loss(fake_feature, real_feature)
        return perceptual_penalty

    def set_ftr_num(self, ftr_num):
        pass

### from dgp NOTE
class DiscriminatorLoss(object):

    def __init__(self, ftr_num=4, data_parallel=False):
        self.l2 = nn.MSELoss()
        self.data_parallel = data_parallel
        self.ftr_num = ftr_num

    def __call__(self, D, fake_img, real_img):
        # import pdb; pdb.set_trace()
        if self.data_parallel:
            with torch.no_grad():
                d, real_feature = nn.parallel.data_parallel(
                    D, real_img.detach())
            d, fake_feature = nn.parallel.data_parallel(D, fake_img)
        else:
            with torch.no_grad():
                d, real_feature = D(real_img.detach())
            d, fake_feature = D(fake_img)
        # shapes of real_feature:
        # torch.Size([1, 96, 64, 64]), torch.Size([1, 96, 64, 64]), torch.Size([1, 192, 32, 32]), torch.Size([1, 384, 16,
        # 16]), torch.Size([1, 768, 8, 8]), torch.Size([1, 1536, 4, 4]), torch.Size([1, 1536, 4, 4]), torch.Size([1, 1536])
        D_penalty = F.l1_loss(fake_feature, real_feature)
        # D_penalty = 0
        # for i in range(self.ftr_num):
        #     f_id = -i - 1
        #     D_penalty = D_penalty + F.l1_loss(fake_feature[f_id],
        #                                       real_feature[f_id])
        return D_penalty

    def set_ftr_num(self, ftr_num):
        self.ftr_num = ftr_num


class ChamferLoss(object):

    def __init__(self):
        self.cd = chamferDist()

    def __call__(self, fake_img, real_img):
        dist1, dist2 = self.cd(fake_img, real_img)
        cd = dist1.mean()+ dist2.mean()
        # import pdb; pdb.set_trace()
        return cd

class EMDLoss(object):
    def __init__(self):
        self.emd = EMD()
    
    def __call__(self, fake_img,real_img):
        # import pdb; pdb.set_trace()
        cost = self.emd(fake_img, real_img)
        return cost
