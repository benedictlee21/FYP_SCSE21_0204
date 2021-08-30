import torch.nn as nn
import torch
import sys
import os
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/emd"))
import emd_module as emd
sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
chamLoss = dist_chamfer_3D.chamfer_3DDist()
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2


class FullModelPCN(nn.Module):
    def __init__(self, model):
        super(FullModelPCN, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, inputs, gt, eps, iters, emd=True, cd=True):
        output1, output2 = self.model(inputs)
        gt = gt[:, :, :3]

        emd1 = emd2 = cd_p1 = cd_p2 = cd_t1 = cd_t2 = torch.tensor([0], dtype=torch.float32).cuda()

        if emd:
            num_coarse = self.model.num_coarse
            gt_fps = pn2.gather_operation(gt.transpose(1, 2).contiguous(),
                                          pn2.furthest_point_sample(gt, num_coarse)).transpose(1, 2).contiguous()

            dist1, _ = self.EMD(output1, gt_fps, eps, iters)
            emd1 = torch.sqrt(dist1).mean(1)

            dist2, _ = self.EMD(output2, gt, eps, iters)
            emd2 = torch.sqrt(dist2).mean(1)

        if cd:
            dist11, dist12, _, _ = chamLoss(gt, output1)
            cd_p1 = (torch.sqrt(dist11).mean(1) + torch.sqrt(dist12).mean(1)) / 2
            cd_t1 = (dist11.mean(1) + dist12.mean(1))

            dist21, dist22, _, _ = chamLoss(gt, output2)
            cd_p2 = (torch.sqrt(dist21).mean(1) + torch.sqrt(dist22).mean(1)) / 2
            cd_t2 = (dist21.mean(1) + dist22.mean(1))
        return output1, output2, emd1, emd2, cd_p1, cd_p2, cd_t1, cd_t2


class FullModelTopNet(nn.Module):
    def __init__(self, model):
        super(FullModelTopNet, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, inputs, gt, eps, iters, emd=True, cd=True):
        output = self.model(inputs)
        gt = gt[:, :, :3]

        emd2 = cd_p = cd_t = torch.tensor([0], dtype=torch.float32).cuda()

        if emd:
            dist, _ = self.EMD(output, gt, eps, iters)

            emd2 = torch.sqrt(dist).mean(1)

        if cd:
            dist1, dist2, _, _ = chamLoss(gt, output)
            cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1))/2
            cd_t = dist1.mean(1) + dist2.mean(1)

        return output, emd2, cd_p, cd_t


class FullModelMSN(nn.Module):
    def __init__(self, model):
        super(FullModelMSN, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, partial, gt, eps, iterations, emd=True, cd=True):
        coarse, fine, expansion_penalty = self.model(partial)
        gt = gt[:, :, :3]

        emd1_out = emd2_out = cd1_p_out = cd2_p_out = cd1_t_out = cd2_t_out = torch.tensor([0],
                                                                                           dtype=torch.float32).cuda()

        if emd:
            dist, _ = self.EMD(coarse, gt, eps, iterations)
            emd1_out = torch.sqrt(dist).mean(1)

            dist, _ = self.EMD(fine, gt, eps, iterations)
            emd2_out = torch.sqrt(dist).mean(1)

        if cd:
            dist1, dist2, _, _ = chamLoss(gt, coarse)
            cd1_p_out = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
            cd1_t_out = dist1.mean(1) + dist2.mean(1)

            dist1, dist2, _, _ = chamLoss(gt, fine)
            cd2_p_out = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
            cd2_t_out = dist1.mean(1) + dist2.mean(1)

        return coarse, fine, emd1_out, emd2_out, cd1_p_out, cd2_p_out, cd1_t_out, cd2_t_out, expansion_penalty


class FullModelCascade(nn.Module):
    def __init__(self, model):
        super(FullModelCascade, self).__init__()
        self.model = model
        self.EMD = emd.emdModule()

    def forward(self, partial, gt, mean_features, eps, iterations, emd=True, cd=True):
        coarse, fine = self.model(partial, mean_features)
        gt = gt[:, :, :3]

        emd1_out = emd2_out = cd1_p_out = cd2_p_out = cd1_t_out = cd2_t_out = torch.tensor([0],
                                                                                           dtype=torch.float32).cuda()

        if emd:
            num_coarse = coarse.shape[1] * 2
            gt_fps = pn2.gather_operation(gt.transpose(1, 2).contiguous(),
                                          pn2.furthest_point_sample(gt, num_coarse)).transpose(1, 2).contiguous()

            if coarse.shape[1] < 1024:
                dist, _ = self.EMD(coarse.repeat(1, 2, 1), gt_fps, eps, iterations)
            else:
                dist, _ = self.EMD(coarse, gt_fps, eps, iterations)

            emd1_out = torch.sqrt(dist).mean(1)

            dist, _ = self.EMD(fine, gt, eps, iterations)
            emd2_out = torch.sqrt(dist).mean(1)

        if cd:
            dist1, dist2, _, _ = chamLoss(gt, coarse)
            cd1_p_out = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
            cd1_t_out = dist1.mean(1) + dist2.mean(1)

            dist1, dist2, _, _ = chamLoss(gt, fine)
            cd2_p_out = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
            cd2_t_out = dist1.mean(1) + dist2.mean(1)

        return coarse, fine, emd1_out, emd2_out, cd1_p_out, cd2_p_out, cd1_t_out, cd2_t_out
