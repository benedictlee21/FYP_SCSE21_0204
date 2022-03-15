import os
import os.path as osp
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from model.treegan_network import Generator, Discriminator
from utils.common_utils import *
from loss import *
from evaluation.pointnet import *
import time
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw

class ShapeInversion(object):

    def __init__(self, args, classes_chosen):
        self.args = args
        print('shape_inversion.py: __init__ - device used:', self.args.device)
        
        self.classes_chosen = classes_chosen
        
        # Check if it is operating in single or multiclass mode based on the class range selected.
        if classes_chosen is not None:
            print('shape_inversion.py: __init__ classes chosen:', self.classes_chosen)
            
        # State the total number of possible classes that can be used for multiclass.
        self.total_num_classes = 8

        if self.args.dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        # init seed for static masks: ball_hole, knn_hole, voxel_mask
        self.to_reset_mask = True
        self.mask_type = self.args.mask_type
        self.update_G_stages = self.args.update_G_stages
        self.iterations = self.args.iterations
        self.G_lrs = self.args.G_lrs
        self.z_lrs = self.args.z_lrs
        self.select_num = self.args.select_num
        self.loss_log = []
        self.latent_space_dim = 96

        # Create the model including generator and discriminator.
        self.G = Generator(features = args.G_FEAT, degrees = args.DEGREE, support = args.support, total_num_classes = self.total_num_classes, args = self.args).cuda()
        self.D = Discriminator(features = args.D_FEAT, total_num_classes = self.total_num_classes, args = self.args).cuda()

        # Define the optimizer and its parameters.
        self.G.optim = torch.optim.Adam(
            [{'params': self.G.get_params(i)}
                for i in range(7)],
            lr=self.G_lrs[0],
            betas=(0,0.99),
            weight_decay=0,
            eps=1e-8)
        
        # Generate the latent space representation.
        # Concatentation of multiclass labels is done in 'treegan_network.py'.
        self.z = torch.zeros((1, 1, self.latent_space_dim)).normal_().cuda()
        
        # 'Variable' is a wrapper for the tensor.
        # Gradients must be computed for the latent space representation tensor.
        self.z = Variable(self.z, requires_grad = True)
        
        # Input to optimizer should be of type 'Variable'.
        self.z_optim = torch.optim.Adam([self.z], lr = self.args.z_lrs[0], betas = (0,0.99))

        # Load pretrained weights from checkpoint, returns a dictionary.
        checkpoint = torch.load(args.ckpt_load, map_location=self.args.device)
        
        # Checkpoint key names: epoch, D_state_dict, G_state_dict, D_loss, G_loss, FPD.
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])

        # Parse the expression arguments for the generator and discriminator.
        # Evaluate those arguments as python expressions.
        self.G.eval()
        if self.D is not None:
            self.D.eval()
            
        # Make a copy of the checkpoint weights to be resued when resetting the generator.
        self.G_weight = deepcopy(self.G.state_dict())

        # Prepare the latent variable and optimizer.
        self.G_scheduler = LRScheduler(self.G.optim, self.args.warm_up)
        self.z_scheduler = LRScheduler(self.z_optim, self.args.warm_up)

        # Define the loss functions.
        self.ftr_net = self.D
        self.criterion = DiscriminatorLoss()

        if self.args.directed_hausdorff:
            self.directed_hausdorff = DirectedHausdorff()

        # For visualization, create a list for holding point cloud stages and their plot titles.
        self.checkpoint_pcd = []
        self.checkpoint_flags = []

        if len(args.w_D_loss) == 1:
            self.w_D_loss = args.w_D_loss * len(args.G_lrs)
        else:
            self.w_D_loss = args.w_D_loss

    # Reset the generator weights after completing each shape.
    def reset_G(self, pcd_id = None):
        """
        to call in every new fine_tuning
        before the 1st one also okay
        """
        self.G.load_state_dict(self.G_weight, strict=False)
        if self.args.random_G:
            self.G.train()
        else:
            self.G.eval()
        self.checkpoint_pcd = [] # to save the staged checkpoints
        self.checkpoint_flags = []
        self.pcd_id = pcd_id # for
        if self.mask_type == 'voxel_mask':
            self.to_reset_mask = True # reset hole center for each shape

    def set_target(self, ground_truth = None, partial = None):
        '''
        set target
        '''
        if ground_truth is not None:
            self.ground_truth = ground_truth.unsqueeze(0)
            # for visualization
            self.checkpoint_flags.append('Ground Truth')
            self.checkpoint_pcd.append(self.ground_truth)
        else:
            self.ground_truth = None

        if partial is not None:
            if self.args.target_downsample_method.lower() == 'fps':
                target_size = self.args.target_downsample_size
                self.target = self.downsample(partial.unsqueeze(0), target_size)
            else:
                self.target = partial.unsqueeze(0)
        
        # If there is no partial shape, transform the input ground truth into a partial shape
        # by degrading it and transforming it in the observation space according to the mask type.
        else:
            self.target = self.pre_process(self.ground_truth, stage = -1)
        # for visualization
        self.checkpoint_flags.append('Target')
        self.checkpoint_pcd.append(self.target)

    def run(self, ith = -1, single_class_id = None):

        loss_dict = {}
        curr_step = 0
        count = 0
        
        # If multiclass with conditional GAN is used, create the one hot encoded
        # generator and discriminator class labels.
        if self.args.class_choice == 'multiclass' and self.args.conditional_gan and single_class_id is not None:
        
            # Create a list of class IDs containing only a single value for completing the input shapes
            # according to a specific class. Number of elements in the list should match the batch size.
            class_id_list = []
            
            for count in range(batch_size):
                class_id_list.append(single_class_id)
            print('Batch of class IDs for testing:', class_id_list)
            
            # Convert the random class ID list into a tensor and place it on the GPU.
            class_id_list = torch.LongTensor(class_id_list)
            class_id_list = random_class_id_list.to(args.device)
        
            one_hot_all_classes = functional.one_hot(class_id_list, num_classes = 8)
            #print('One hot encoded classes:', one_hot_all_classes)
            #print('One hot encoded classes shape:', one_hot_all_classes.shape)
            
            # Convert the tensor data type to 'long' and move it to the GPU.
            discriminator_class_labels = torch.LongTensor(one_hot_all_classes)
            discriminator_class_labels.to(self.args.device)
            
            # Resultant discriminator class labels shape should be: <batch size, <number of classes>.
            #print('Shape of discriminator class labels:', discriminator_class_labels.shape)
            #print('One hot encoded discriminator class labels:', discriminator_class_labels)
            
            generator_class_labels = torch.LongTensor(one_hot_all_classes)
            generator_class_labels.to(self.args.device)
            generator_class_labels = torch.unsqueeze(generator_class_labels, 1)
            
            # Resultant generator class labels shape should be: <batch size, 1, <number of classes>.
            #print('Shape of generator class labels:', generator_class_labels.shape)
            
            # Resultant discriminator class labels shape should be: <batch size, <number of classes>.
            #print('Shape of discriminator class labels:', discriminator_class_labels.shape)
            #print('One hot encoded discriminator class labels:', discriminator_class_labels)
            
        # For each shape.
        for stage, iteration in enumerate(self.iterations):

            # For each step in the shape completion process.
            for i in range(iteration):
                curr_step += 1

                # Reset the learning rate parameters based on input arguments.
                self.G_scheduler.update(curr_step, self.args.G_lrs[stage])
                self.z_scheduler.update(curr_step, self.args.z_lrs[stage])

                # Reset all latent space optimizer model weights to zero.
                self.z_optim.zero_grad()

                # If updating the generator stage, reset all generator optimizer weights to zero.
                if self.update_G_stages[stage]:
                    self.G.optim.zero_grad()

                # Store the latent space into a list and pass it to the generator.
                tree = [self.z]

                # Pass the latent space representation to 'treegan.py' where one hot encoding is performed.
                x = self.G(tree, generator_class_labels, self.args.device)

                # Perform degradation of generated shape and masking, where 'x' is the generated shape.
                x_map = self.pre_process(x, stage = stage)

                # Compute losses.
                ftr_loss = self.criterion(self.ftr_net, x_map, self.target)
                dist1, dist2 , _, _ = distChamfer(x_map, self.target)
                cd_loss = dist1.mean() + dist2.mean()

                # Perform optional early stopping if specified as an argument.
                if self.args.early_stopping:
                
                    # Stop based on the defined chamfer distance value threshold.
                    if cd_loss.item() < self.args.stop_cd:
                        break

                # 'nll' corresponds to a negative log-likelihood loss.
                nll = self.z**2 / 2
                nll = nll.mean()

                # Compute loss.
                loss = ftr_loss * self.w_D_loss[stage] + nll * self.args.w_nll \
                        + cd_loss * 1

                # Use optional directed_hausdorff if specified as an arugment.
                if self.args.directed_hausdorff:
                    directed_hausdorff_loss = self.directed_hausdorff(self.target, x)
                    loss += directed_hausdorff_loss*self.args.w_directed_hausdorff_loss

                # Perform backward propagation.
                loss.backward()
                self.z_optim.step()
                if self.update_G_stages[stage]:
                    self.G.optim.step()

            # Save checkpoint and its label for each stage.
            self.checkpoint_flags.append('s_'+str(stage)+' x')
            self.checkpoint_pcd.append(x)
            self.checkpoint_flags.append('s_'+str(stage)+' x_map')
            self.checkpoint_pcd.append(x_map)

            # If the ground truth exists, calculate the chamfer distance for each shape.
            if self.ground_truth is not None:
                dist1, dist2 , _, _ = distChamfer(x,self.ground_truth)
                test_cd = dist1.mean() + dist2.mean()

                # Log the results.
                with open(self.args.log_pathname, "a") as file_object:
                    msg = str(self.pcd_id) + ',' + 'stage'+str(stage) + ',' + 'cd' +',' + '{:6.5f}'.format(test_cd.item())
                    file_object.write(msg+'\n')

        # If the ground truth exists, determine additional loss values.
        if self.ground_truth is not None:
            loss_dict = {
                'ftr_loss': np.asscalar(ftr_loss.detach().cpu().numpy()),
                'nll': np.asscalar(nll.detach().cpu().numpy()),
                'cd': np.asscalar(test_cd.detach().cpu().numpy()),
            }
            self.loss_log.append(loss_dict)

        # Save the completed point cloud output.
        self.x = x
       
       # Create the path to save the results if it does not exist.
        if not osp.isdir(self.args.save_inversion_path):
            os.mkdir(self.args.save_inversion_path)
        
        # Get the input partial shape, its map and the completed shape as numpy arrays.
        x_np = x[0].detach().cpu().numpy()
        x_map_np = x_map[0].detach().cpu().numpy()
        target_np = self.target[0].detach().cpu().numpy()

        if ith == -1:
            basename = str(self.pcd_id)
        else:
            basename = str(self.pcd_id)+'_'+str(ith)

        # If the ground truth exists, save it to the output.
        if self.ground_truth is not None:
            ground_truth_np = self.ground_truth[0].detach().cpu().numpy()
            np.savetxt(osp.join(self.args.save_inversion_path,basename+'_Ground_Truth.txt'), ground_truth_np, fmt = "%f;%f;%f")
            
        # Save the input partial, its map and the completed shape.
        np.savetxt(osp.join(self.args.save_inversion_path,basename+'_X_Partial_Shape.txt'), x_np, fmt = "%f;%f;%f")
        np.savetxt(osp.join(self.args.save_inversion_path,basename+'_X_Partial_Shape_Map.txt'), x_map_np, fmt = "%f;%f;%f")
        np.savetxt(osp.join(self.args.save_inversion_path,basename+'_Completed_Shape.txt'), target_np, fmt = "%f;%f;%f")

        # If the shapeinversion mode is 'jittering'.
        if self.args.inversion_mode == 'jittering':
            self.jitter(self.target)

    # Search for latent space gaussian distributions for each completed shape of 'diversity' mode.
    def diversity_search(self):
        """
        produce batch by batch
        search by 2pf and partial
        but constrainted to z dimension are large
        """
        batch_size = 1

        num_batch = int(self.select_num/batch_size)
        x_ls = []
        z_ls = []
        cd_ls = []
        tic = time.time()

        # Reset the gradients and pass the latent space representation to the generator.
        with torch.no_grad():
        
            # For each batch.
            for i in range(num_batch):
            
                # Generate the latent space vector tensor.
                z = torch.randn(batch_size, 1, self.latent_space_dim).cuda()
                tree = [z]
                x = self.G(tree)
                
                # Compute the chamfer distance.
                dist1, dist2 , _, _ = distChamfer(self.target.repeat(batch_size,1,1),x)
                        
                # Compute single directional chamfer distance.
                cd = dist1.mean(1)

                # Append the generated shape, latent space and chamfer distance to their respective lists.
                x_ls.append(x)
                z_ls.append(z)
                cd_ls.append(cd)

        x_full = torch.cat(x_ls)
        cd_full = torch.cat(cd_ls)
        z_full = torch.cat(z_ls)
        toc = time.time()

        cd_candidates, idx = torch.topk(cd_full,self.args.n_z_candidates,largest=False)
        z_t = z_full[idx].transpose(0,1)
        seeds = farthest_point_sample(z_t, self.args.n_outputs).squeeze(0)
        z_ten = z_full[idx][seeds]
        self.zs = [itm.unsqueeze(0) for itm in z_ten]
        self.xs = []

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
            for i in range(self.select_num):
                z = torch.randn(1, 1, latent_space_dim).cuda()
                tree = [z]
                with torch.no_grad():
                    x = self.G(tree)
                ftr_loss = self.criterion(self.ftr_net, x, self.target)
                z_all.append(z)
                loss_all.append(ftr_loss.detach().cpu().numpy())

            toc = time.time()
            loss_all = np.array(loss_all)
            idx = np.argmin(loss_all)

            self.z.copy_(z_all[idx])
            if select_y:
                self.y.copy_(y_all[idx])

            x = self.G([self.z])

            # If the ground truth exists, perform degradation of the generated shape and
            # evaluate it against the ground truth using chamfer distance.
            if self.ground_truth is not None:
                x_map = self.pre_process(x, stage=-1)
                dist1, dist2 , _, _ = distChamfer(x,self.ground_truth)
                cd_loss = dist1.mean() + dist2.mean()

                # Log the results.
                with open(self.args.log_pathname, "a") as file_object:
                    msg = str(self.pcd_id) + ',' + 'init' + ',' + 'cd' +',' + '{:6.5f}'.format(cd_loss.item())
                    # print(msg)
                    file_object.write(msg+'\n')
                self.checkpoint_flags.append('init x')
                self.checkpoint_pcd.append(x)
                self.checkpoint_flags.append('init x_map')
                self.checkpoint_pcd.append(x_map)
            return z_all[idx]


    def pre_process(self, pcd, stage = -1):
        """
        transfer a pcd in the observation space:
        with the following mask_type:
            none: for ['reconstruction', 'jittering', 'morphing']
            ball_hole, knn_hole: randomly create the holes from complete pcd, similar to PF-Net
            voxel_mask: baseline in ShapeInversion
            tau_mask: baseline in ShapeInversion
            k_mask: proposed component by ShapeInversion
        """

        # 'pcd' represents the ground truth.
        if self.mask_type == 'none':
            return pcd
            
        elif self.mask_type in ['ball_hole', 'knn_hole']:
            ### set static mask for each new partial pcd
            if self.to_reset_mask:
                # either ball hole or knn_hole, hence there might be unused configs
                self.hole_k = self.args.hole_k
                self.hole_radius = self.args.hole_radius
                self.hole_n = self.args.hole_n
                seeds = farthest_point_sample(pcd, self.hole_n) # shape (B,hole_n)
                self.hole_centers = torch.stack([img[seed] for img, seed in zip(pcd,seeds)]) # (B, hole_n, 3)
                # turn off mask after set mask, until next partial pcd
                self.to_reset_mask = False

            ### preprocess
            flag_map = torch.ones(1,2048,1).cuda()
            pcd_new = pcd.unsqueeze(2).repeat(1,1,self.hole_n,1)
            seeds_new = self.hole_centers.unsqueeze(1).repeat(1,2048,1,1)
            delta = pcd_new.add(-seeds_new) # (B, 2048, hole_n, 3)
            dist_mat = torch.norm(delta,dim=3)
            dist_new = dist_mat.transpose(1,2) # (B, hole_n, 2048)

            if self.mask_type == 'knn_hole':
                # idx (B, hole_n, hole_k), dist (B, hole_n, hole_k)
                dist, idx = torch.topk(dist_new,self.hole_k,largest=False)

            for i in range(self.hole_n):
                dist_per_hole = dist_new[:,i,:].unsqueeze(2)
                if self.mask_type == 'knn_hole':
                    threshold_dist = dist[:,i, -1]
                if self.mask_type == 'ball_hole':
                    threshold_dist = self.hole_radius
                flag_map[dist_per_hole <= threshold_dist] = 0

            target = torch.mul(pcd, flag_map)
            return target
            
        elif self.mask_type == 'voxel_mask':
            """
            voxels in the partial and optionally surroundings are 1, the rest are 0.
            """
            ### set static mask for each new partial pcd
            if self.to_reset_mask:
                mask_partial = self.voxelize(self.target, n_bins=self.args.voxel_bins, pcd_limit=0.5, threshold=0)
                # optional to add surrounding to the mask partial
                surrounding = self.args.surrounding
                self.mask_dict = {}
                
                for key_ground_truth in mask_partial:
                    x,y,z = key_ground_truth
                    surrounding_ls = []
                    surrounding_ls.append((x,y,z))
                    
                    for x_s in range(x-surrounding+1, x+surrounding):
                    
                        for y_s in range(y-surrounding+1, y+surrounding):
                        
                            for z_s in range(z-surrounding+1, z+surrounding):
                                surrounding_ls.append((x_s,y_s,z_s))
                                
                    for xyz in surrounding_ls:
                        self.mask_dict[xyz] = 1
                # turn off mask after set mask, until next partial pcd
                self.to_reset_mask = False

            ### preprocess
            n_bins = self.args.voxel_bins
            mask_tensor = torch.zeros(2048,1)
            pcd_new = pcd*n_bins + n_bins * 0.5
            pcd_new = pcd_new.type(torch.int64)
            ls_voxels = pcd_new.squeeze(0).tolist() # 2028 of sublists
            tuple_voxels = [tuple(itm) for itm in ls_voxels]
            
            for i in range(2048):
                tuple_voxel = tuple_voxels[i]
                if tuple_voxel in self.mask_dict:
                    mask_tensor[i] = 1

            mask_tensor = mask_tensor.unsqueeze(0).cuda()
            pcd_map = torch.mul(pcd, mask_tensor)
            return pcd_map
            
        # 'self.target' represents either the input partial shape or the ground truth degraded into a partial shape.
        # 'pcd' represents the ground truth.
        elif self.mask_type == 'k_mask':
            
            pcd_map = self.k_mask(self.target, pcd, stage)
            return pcd_map
            
        elif self.mask_type == 'tau_mask':
            pcd_map = self.tau_mask(self.target, pcd,stage)
            return pcd_map
            
        else:
            raise NotImplementedError

    def voxelize(self, pcd, n_bins=32, pcd_limit=0.5, threshold=0):
        """
        given a partial/ground truth pcd
        return {0,1} masks with resolution n_bins^3
        voxel_limit in case the pcd is very small, but still assume it is symmetric
        threshold is needed, in case we would need to handle noise
        the form of output is a dict, key (x,y,z) , value: count
        """
        pcd_new = pcd * n_bins + n_bins * 0.5
        pcd_new = pcd_new.type(torch.int64)
        ls_voxels = pcd_new.squeeze(0).tolist() # 2028 of sublists
        tuple_voxels = [tuple(itm) for itm in ls_voxels]
        mask_dict = {}
        for tuple_voxel in tuple_voxels:
            if tuple_voxel not in mask_dict:
                mask_dict[tuple_voxel] = 1
            else:
                mask_dict[tuple_voxel] += 1
        for voxel, cnt in mask_dict.items():
            if cnt <= threshold:
                del mask_dict[voxel]
        return mask_dict

    # Perform masking based on tau distance that uses the original chamfer distance algorithm.
    # Corresponding matching points and regions in both input partial and the generated complete
    # shape are determined based on the distance from their nearest neighbour using a predefined threshold.
    def tau_mask(self, target, x, stage = -1):

        # dist_mat shape (B, N_target, N_output), where B = 1
        stage = max(0, stage)
        
        # Get the tau distance threshold for the tau mask.
        dist_tau = self.args.tau_mask_dist[stage]
        
        # Compute the chamfer distance between the partial and ground truth.
        # Returns the indexes of the closest point on shape A from the points of shape B.
        dist_mat = distChamfer_raw(target, x)
        
        # Filter and keep only the point indexes where the chamfer distances are greater than the tau distances.
        idx0, idx1, idx2 = torch.where(dist_mat < dist_tau)
        
        # Get the unique index elements from 'idx2' and convert the values to type 'long'.
        idx = torch.unique(idx2).type(torch.long)
        
        # Obtain a map points consisting of only those unique index values.
        x_map = x[:, idx]
        return x_map

    # Perform K mask masking based on chamfer distance which was developed by ShapeInversion authors.
    def k_mask(self, target, x, stage = -1):
        """
        target: (1, N, 3), partial, can be < 2048, 2048, > 2048
        x: (1, 2048, 3) ground truth
        x_map: (1, N', 3), N' < 2048
        x_map: v1: 2048, 0 masked points
        """
        # Select the index to retrieve the k mask value.
        stage = max(0, stage)
        
        # Retrieve the k mask value from the array.
        knn = self.args.k_mask_k[stage]
        if knn == 1:
        
            # Compute the chamfer distance.
            cd1, cd2, argmin1, argmin2 = distChamfer(target, x)
            
            # Get the unique elements from the tensor 'indices' and convert the values to type 'long'.
            # union of all the indices
            idx = torch.unique(argmin1).type(torch.long)
            
        elif knn > 1:
            # Compute the chamfer distance between the partial and ground truth.
            # Returns the indexes of the closest point on shape A from the points of shape B.
            # dist_mat shape (B, 2048, 2048), where B = 1
            dist_mat = distChamfer_raw(target, x)
            
            # Get the 'k' smallest elements from the tensor 'dist_mat' along dimension = 2.
            # indices (B, 2048, k)
            val, indices = torch.topk(dist_mat, k = knn, dim = 2, largest = False)
            
            # Get the unique elements from the tensor 'indices' and convert the values to type 'long'.
            idx = torch.unique(indices).type(torch.long)

        # Keep the zeros with the element product.
        if self.args.masking_option == 'element_product':
            
            # Create a tensor of zeros.
            mask_tensor = torch.zeros(2048, 1)
            
            # Set the earlier identified unique index values to 1.
            mask_tensor[idx] = 1
            
            # Add a dimension of size 1 to the tensor at position 0.
            mask_tensor = mask_tensor.cuda().unsqueeze(0)
            
            # Multiply the mask with the ground truth to obtain the partial shape.
            # Similar to applying a per pixel binary mask to a 2D image using AND operation.
            x_map = torch.mul(x, mask_tensor)
            
        # Remove zeros from the element product.
        elif self.args.masking_option == 'indexing':
            x_map = x[:, idx]
        return x_map

    def jitter(self, x):
        z_rand = self.z.clone()

        stds = [0.3, 0.5, 0.7]
        n_jitters = 12

        flag_list = ['ground_truth', 'recon']
        pcd_list = [self.ground_truth, self.x]
        with torch.no_grad():
            for std in stds:

                for i in range(n_jitters):
                    z_rand.normal_()
                    z = self.z + std * z_rand
                    x_jitter = self.G([z])
                    x_np = x_jitter.squeeze(0).detach().cpu().numpy()
                    basename = '{}_std{:3.2f}_{}.txt'.format(self.pcd_id,std,i)
                    pathname = osp.join(self.args.save_inversion_path,basename)
                    np.savetxt(pathname, x_np, fmt = "%f;%f;%f")
                    flag_list.append(basename)
                    pcd_list.append(x_jitter)
        self.checkpoint_pcd = pcd_list
        self.checkpoint_flags = flag_list

    def downsample(self, dense_pcd, n=2048):
        """
        input pcd cpu tensor
        return downsampled cpu tensor
        """
        idx = farthest_point_sample(dense_pcd,n)
        sparse_pcd = dense_pcd[0,idx]
        return sparse_pcd

