import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from data.CRN_dataset import CRNShapeNet
from model.treegan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd, calculate_activation_statistics
from metrics import *
from loss import *
from evaluation.pointnet import PointNetCls
from math import ceil
from utils.common_utils import *
from arguments import Arguments
from encode_classes import encode_classes
import argparse
import time
import numpy as np
import time
import os.path as osp
import os
import copy
import random

def save_pcs_to_txt(save_dir, fake_pcs):
    """
    save pcds into txt files
    """
    sample_size = fake_pcs.shape[0]
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    for i in range(sample_size):
        np.savetxt(osp.join(save_dir,str(i)+'.txt'), fake_pcs[i], fmt = "%f;%f;%f")  

# 'model_cuda' represents the instantiated generator network.
def generate_pcs(model_cuda, n_pcs = 5000, batch_size = 50, device = None, latent_space_dim = 96, total_number_classes = None, classes_chosen = None, args = None):
    """
    generate fake pcs for evaluation
    """
    # Add the number of classes to the latent space dimensions fo multiclass operation.
    if args.class_choice == 'multiclass' and args.conditional_gan:
        latent_space_dim += total_number_classes

    fake_pcs = torch.Tensor([])
    n_pcs = int(ceil(n_pcs/batch_size) * batch_size)
    n_batches = ceil(n_pcs/batch_size)

    for i in range(n_batches):
    
        # Generate the latent space vector tensor.
        z = torch.randn(batch_size, 1, latent_space_dim).to(device)
                
        # Pass the latent space representation to the generator.
        tree = [z]
        
        # Randomly select class IDs to be used for generating the point clouds as per the 
        # selected classes, with the number of randomly selected class IDs equal to the batch size.
        random_class_id_list = []
        
        for count in range(batch_size):
            random_class_id = random.choice(classes_chosen)
            random_class_id_list.append(random_class_id)
        print('Randomly generated batch of class IDs for evaluation:', random_class_id_list)
        
        # Convert the random class ID list into a tensor and place it on the GPU.
        random_class_id_list = torch.LongTensor(random_class_id_list)
        random_class_id_list = random_class_id_list.to(args.device)
        
        # Perform the one hot encoding.
        # Conditional GAN argument is of type boolean.
        # No class tensor is created or concatenated if the conditional gan option is false.
        if args.class_choice == 'multiclass' and args.conditional_gan:
            
            # ----------------------------------------------------------------------------
            # Perform one hot encoding on tensor using torch nn functional package.
            # This creates a vector of 8 bits, of which only 1 bit has a value of 1,
            # corresponding to the class for that shape being retrieved.
            
            one_hot_all_classes = functional.one_hot(random_class_id_list, num_classes = 8)
            #print('One hot encoded classes:', one_hot_all_classes)
            #print('One hot encoded classes shape:', one_hot_all_classes.shape)
            
            # Move the one hot tensor to the cpu and convert its type to 'long'.
            # Then move it back to the GPU.
            one_hot_all_classes = one_hot_all_classes.cpu()
            discriminator_class_labels = torch.LongTensor(one_hot_all_classes)
            discriminator_class_labels = discriminator_class_labels.to(args.device)
                    
            # Resultant discriminator class labels shape should be: <batch size, <number of classes>.
            #print('Shape of discriminator class labels:', discriminator_class_labels.shape)
            #print('One hot encoded discriminator class labels:', discriminator_class_labels)
            
            # Move the one hot tensor to the cpu and convert its type to 'long'.
            # Then move it back to the GPU.
            generator_class_labels = torch.LongTensor(one_hot_all_classes)
            generator_class_labels = torch.unsqueeze(generator_class_labels, 1)
            generator_class_labels = generator_class_labels.to(args.device)
            
            # Resultant generator class labels shape should be: <batch size, 1, <number of classes>.
            #print('Shape of generator class labels:', generator_class_labels.shape)
            # ----------------------------------------------------------------------------
        else:
            generator_class_labels = None
            discriminator_class_labels = None
                
        # Reset the gradients and pass the latent space representation to the generator.
        with torch.no_grad():
            sample = model_cuda(tree, generator_class_labels).cpu()
            
        # Concatenate the completed shapes to the input shape.
        fake_pcs = torch.cat((fake_pcs, sample), dim = 0)
        
    return fake_pcs

def create_fpd_stats(pcs, pathname_save, device):
    """
    create stats of training data for evaluation of fpd
    """
    PointNet_pretrained_path = './evaluation/cls_model_39.pth'

    model = PointNetCls(k=16).to(device)
    model.load_state_dict(torch.load(PointNet_pretrained_path))
    mu, sigma = calculate_activation_statistics(pcs, model, device=device)
    print (mu.shape, sigma.shape)
    np.savez(pathname_save,m=mu,s=sigma)
    print('FPD statistics saved to:', pathname_save)

@timeit
def script_create_fpd_stats(args, total_number_classes = None, classes_chosen = None, data2stats = 'CRN'):
    """
    create stats of training data for eval FPD, calling create_fpd_stats()
    """    
    if data2stats == 'CRN':
    
        # 'is_eval' argument specifies to draw all shapes from the test dataset.
        # 'is_eval' is only flase when pretraining using a smaller sample size from the training dataset.
        dataset = CRNShapeNet(args, classes_chosen = classes_chosen, is_eval = True)
        
        # Retrieve the data from the dataset for evaluation.
        dataLoader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 8)

        # Preparation for naming of NPZ statistics file for multiclass.
        if args.class_choice == 'multiclass':
            class_list = args.class_range.split(',')
            
            # Concatenate all the class names into a string.
            multiclass_string = ''
            
            for index in range(len(class_list)):
                multiclass_string = multiclass_string + class_list[index]
                
                # Continue adding the underscore between classes.
                if index + 1 < len(class_list):
                    multiclass_string = multiclass_string + '_'

            # Generate the full filename.
            pathname_save = './evaluation/pre_statistics_CRN_' + args.class_choice + '_' + multiclass_string + '.npz'
        
        # For single class, just use the class name.
        else:
            pathname_save = './evaluation/pre_statistics_CRN_' + args.class_choice + '.npz'

    else:
        raise NotImplementedError

    ref_pcs = torch.Tensor([])
    
    # Returns shapes and their class ID one batch at a time.
    for _iter, data in enumerate(dataLoader):
        
        # 'point' is the tensor containing ground truths.
        # '_' means to ignore that return variable.
        point, _, _, _ = data
        ref_pcs = torch.cat((ref_pcs, point),0)
    
    create_fpd_stats(ref_pcs,pathname_save, args.device)
    
@timeit
def checkpoint_eval(G_net, device, n_samples = 5000, batch_size = 100,conditional = False, ratio = 'even', FPD_path = None, class_choices = None, latent_space_dim = 96):
    """
    an abstraction used during training
    """
    G_net.eval()
    fake_pcs = generate_pcs(G_net, n_pcs = n_samples, batch_size = batch_size, device = device, latent_space_dim = latent_space_dim)
    fpd = calculate_fpd(fake_pcs, statistic_save_path = FPD_path, batch_size = 100, dims = 1808, device = device)
    # print(fpd)
    print('----------------------------------------- Frechet Pointcloud Distance <<< {:.2f} >>>'.format(fpd))

@timeit
def test(args, mode = 'FPD', total_number_classes = None, classes_chosen = None):
    '''
    args needed: 
        n_classes, pcs to generate, ratio of each class, class to id dict???
        model pth, , points to save, save pth, npz for the class, 
    '''
    # Extract the generator features first in case it needs to be modified for multiclass.
    generator_features = args.G_FEAT
    discriminator_features = args.D_FEAT
    
    # Instantiate an instance of the generator.
    G_net = Generator(features = generator_features, degrees = args.DEGREE, support = args.support, total_num_classes = total_number_classes, args = args).to(args.device)
    
    # Instantiate an instance of the discriminator only for multiclass to update the latent space dimensions.
    # Discriminator object is not used for the evaluation process.
    D_net = Discriminator(features = discriminator_features, total_num_classes = total_number_classes, args = args).to(args.device)

    # Loading of specified model checkpoint.
    checkpoint = torch.load(args.model_pathname, map_location=args.device)
    G_net.load_state_dict(checkpoint['G_state_dict'])
    G_net.eval()
    
    # Specify the number of latent space dimensions.
    latent_space_dim = 96
    
    # Generate the point cloud shapes using the generator.
    fake_pcs = generate_pcs(G_net, n_pcs = args.n_samples, batch_size = args.batch_size, device = args.device, latent_space_dim = latent_space_dim, total_number_classes = total_number_classes, classes_chosen = classes_chosen, args = args)
    
    # Save generated samples.
    if mode == 'save':
        save_pcs_to_txt(args.save_sample_path, fake_pcs)
        
    # Calculate fractal point distance.
    elif mode == 'FPD':
        fpd = calculate_fpd(fake_pcs, statistic_save_path=args.FPD_path, batch_size=100, dims=1808, device=args.device)
        print('-----FPD: {:3.2f}'.format(fpd))
            
    # Calculate minimum matching distance.
    elif mode == 'MMD':
        use_EMD = True
        batch_size = 50 
        normalize = True
        gt_dataset = CRNShapeNet(args)
        
        dataLoader = torch.utils.data.DataLoader(gt_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=10)
        gt_data = torch.Tensor([])
        
        # For each batch of shapes, retrieve the ground truth, partial and shape index.
        # Shape class ID is not required.
        for _iter, data in enumerate(dataLoader):
            point, partial, index, _ = data
            gt_data = torch.cat((gt_data,point),0)
        ref_pcs = gt_data.detach().cpu().numpy()
        sample_pcs = fake_pcs.detach().cpu().numpy()

        tic = time.time()
        mmd, matched_dists, dist_mat = MMD_batch(sample_pcs,ref_pcs,batch_size, normalize=normalize, use_EMD=use_EMD,device=args.device)
        toc = time.time()
        print('-----MMD-EMD: {:5.3f}'.format(mmd*100))

if __name__ == '__main__':
    args = Arguments(stage='eval_treegan').parser().parse_args()
    args.device = torch.device('cuda')

    # Ensure that a valid evalutation mode is specified.
    assert args.eval_treegan_mode in ["MMD", "FPD", "save", "generate_fpd_stats"]
    
    # Convert the range of classes specified by the user as lowercase.
    args.class_choice = args.class_choice.lower()
    
    # If multiclass pretraining is specified.
    if args.class_choice == 'multiclass' and args.class_range is not None:
        
        # Convert the one hot encoding list into an array, representing the classes.
        classes_chosen = encode_classes(args.class_range)
        #print('eval_treegan.py: main - classes chosen:', classes_chosen)
        
    # Otherwise if only using a single class.
    else:
        classes_chosen = None
    
    # Specify the total number of possible classes.
    total_number_classes = 8
    
    if args.eval_treegan_mode == "generate_fpd_stats":
    
        # Generation of FPD statistics file when running multiclass operation already makes use of multiclass
        # intergration implemented in 'CRN_dataset.py', hence no need to provide argument for classes chosen here.
        script_create_fpd_stats(args, total_number_classes)
    else:   
        test(args, mode = args.eval_treegan_mode, total_number_classes = total_number_classes, classes_chosen = classes_chosen)
