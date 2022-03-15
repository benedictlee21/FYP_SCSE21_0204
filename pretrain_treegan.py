import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from data.CRN_dataset import CRNShapeNet
from model.treegan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd
from arguments import Arguments
import time
import numpy as np
from loss import *
from metrics import *
import os
import os.path as osp
from eval_treegan import checkpoint_eval
from encode_classes import encode_classes
from torch.utils.tensorboard import SummaryWriter

class TreeGAN():
    def __init__(self, args):
        
        self.args = args
        self.args.class_choice = self.args.class_choice.lower()
        
        # Create a tensorboard summarywriter object for recording training metrics.
        self.shapeinversion_writer = SummaryWriter()

# --------------------------------------------------------
        # If multiclass pretraining is specified.
        if self.args.class_choice == 'multiclass' and self.args.class_range is not None:
            
            # Convert the one hot encoding list into an array, representing the classes.
            self.classes_chosen = encode_classes(self.args.class_range)
            
            # Set the total number of classes used for the whole experiment. Total of 8 possible classes:
            # chair, table, couch, cabinet, lamp, car, plane, watercraft.
            self.total_num_classes = 8
            
        # Otherwise if only using a single class.
        else:
            self.classes_chosen = None
            self.total_num_classes = 0
            print('pretrain_treegan.py: __init__ - single class pretraining.')
# --------------------------------------------------------

        # Load the dataset. Reduce the number of workers here if the data loading process freezes.
        # Original number of workers is 16.
        # 'self.data' returns a 'CRNShapeNet' object.
        self.data = CRNShapeNet(self.args, self.classes_chosen)
        
        # Append the class index to each shape in the dataset.        
        # Combines the dataset with a sampler, providing an iterable over the dataset.
        # Shuffling of shapes by class indexes is performed here.
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 2)
        
        print("Training Dataset : {} prepared.".format(len(self.data)))

        # Define the generator and discriminator models.
        # Pass in the chosen classes if multiclass is specified.
        self.G = Generator(features = args.G_FEAT, degrees = args.DEGREE, support = args.support, total_num_classes = self.total_num_classes, args = self.args).to(args.device)
        # import pdb; pdb.set_trace()
        self.D = Discriminator(features = args.D_FEAT, total_num_classes = self.total_num_classes, args = self.args).to(args.device)
        
        # Define the optimizer and parameters.
        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))
        
        # Define the calculation of gradient penalty.
        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)

        # Calculate unifrom losses.
        # Expansion penalty.
        if self.args.expansion_penality:
            self.expansion = expansionPenaltyModule()

        # PU-Net using Krepul loss.
        if self.args.krepul_loss:
            self.krepul_loss = kNNRepulsionLoss(k=self.args.krepul_k,n_seeds=self.args.krepul_n_seeds,h=self.args.krepul_h)

        # PatchVariance using knn_loss.
        if self.args.knn_loss:
            self.knn_loss = kNNLoss(k=self.args.knn_k,n_seeds=self.args.knn_n_seeds)

        print("Network prepared.")
        # ----------------------------------------------------------------------------------------------------- #
        if len(args.w_train_ls) == 1:
            self.w_train_ls = args.w_train_ls * 4
        else:
            self.w_train_ls = args.w_train_ls
        

    def run(self, save_ckpt=None, load_ckpt=None):

        epoch_log = 0
        loss_log = {'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())
        metric = {'FPD': []}

        # Load previously saved checkpoint if applicable.
        if load_ckpt is not None:
            checkpoint = torch.load(load_ckpt, map_location=self.args.device)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']
            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())
            metric['FPD'] = checkpoint['FPD']

            print("Checkpoint loaded.")

        # Enable data parallelism after loading.
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)
        
        # Prepare the number of dimensions for the latent space for the network.
        latent_space_dim = 96

        # For each epoch.
        for epoch in range(epoch_log, self.args.epochs):
            print('Starting epoch:', epoch + 1)
            epoch_g_loss = []
            epoch_d_loss = []
            epoch_time = time.time()
            self.w_train = self.w_train_ls[min(3,int(epoch/500))]

            # Load each shape one at a time from the dataloader.
            # Dataloader uses function '__getitem__' in 'CRNShapeNet.py'.
            for _iter, data in enumerate(self.dataLoader):
                
                # Start time.
                start_time = time.time()
                
                # Assigned variables are those returned from the '__getitem__' function in 'CRNShapeNet.py'.
                # 'point' is the tensor containing ground truths.
                # 'class_id' is the tensor of class IDs.
                # '_' means to ignore that respective return variable.
                point, _, _, class_id = data
                point = point.to(self.args.device)
                class_id = class_id.to(self.args.device)
                
                # Number of shapes in ground truth tensor and indexes in class tensor are determined by the batch size.
                #print('Ground truth (point) tensor:', point)
                #print('Ground truth (point) tensor shape:', point.shape)
                #print('Class ID tensor:', class_id)
                #print('Class ID tensor shape:', class_id.shape)
                
# ++++++++++++++++++++++++++++++++++++++++++++++++++
                # Creating zeroed tensor for onehot encoding.
                # zeroed_tensor = torch.zeros(<tensor_dim0_size>, <tensor_dim1_size>, dtype = <data_type>)
                
                # One hot encoding class values using torch.tensor.scatter().
                # onehot_tensor = zeroed_tensor.scatter(<axis along which to index>, <tensor of element positional indices to assign the one hot encoded values to>, <source tensor whose values are to be assigned>)
                
                # One hot encoding by junzhe.
                # NOTE: see how scatter works: 
                # https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_
                                
                # Conditional GAN argument is of type boolean.
                # No class tensor is created or concatenated if the conditional gan option is false.
                if self.args.conditional_gan:
                    
                    # ----------------------------------------------------------------------------
                    # Perform one hot encoding on tensor using torch nn functional package.
                    # This creates a vector of 8 bits, of which only 1 bit has a value of 1,
                    # corresponding to the class for that shape being retrieved.
                    one_hot_all_classes = functional.one_hot(class_id, num_classes = 8).cpu()
                    #print('One hot encoded classes:', one_hot_all_classes)
                    #print('One hot encoded classes shape:', one_hot_all_classes.shape)
                    
                    # Convert the tensor data type to 'long' and move it to the GPU.
                    discriminator_class_labels = torch.LongTensor(one_hot_all_classes)
                    discriminator_class_labels.to(self.args.device)
                    
                    # Resultant discriminator class labels shape should be: <batch size, <number of classes>.
                    #print('Shape of discriminator class labels:', discriminator_class_labels.shape)
                    #print('One hot encoded discriminator class labels:', discriminator_class_labels)
                    # ----------------------------------------------------------------------------
                else:
                    discriminator_class_labels = None
                    
# ++++++++++++++++++++++++++++++++++++++++++++++++++

# -------------------- Discriminator -------------------- #
                tic = time.time()
                
                # Repeat for the number of iterations for the discriminator.
                for d_iter in range(self.args.D_iter):
                    
                    # Reset discriminator gradients to zero.
                    self.D.zero_grad()
                    
                    # Generate the latent space representation.
                    # First dimension of latent space represents the batch size.
                    z = torch.randn(point.shape[0], 1, latent_space_dim).to(self.args.device)

# ++++++++++++++++++++++++++++++++++++++++++++++++++
                    if self.args.conditional_gan:
                    
                        # Convert the tensor data type to 'long' and move it to the GPU.
                        generator_class_labels = torch.LongTensor(one_hot_all_classes)
                        generator_class_labels.to(self.args.device)
                        generator_class_labels = torch.unsqueeze(generator_class_labels, 1)
                        
                        # Resultant generator class labels shape should be: <batch size, 1, <number of classes>.
                        #print('Shape of generator class labels:', generator_class_labels.shape)
                        
                    else:
                        generator_class_labels = None
# ++++++++++++++++++++++++++++++++++++++++++++++++++
                    
                    # Pass the latent space representation to the generator.
                    tree = [z]
                    #print('pretrain_treegan.py - tree[0] shape:', tree[0].size())

                    # Concatenation of latent space with class tensor is performed in 'treegan_network.py'.
                    # Reset the gradients and pass the latent space representation to the generator.
                    # 'self.G' leads into the 'forward()' function of the generator in 'treegan_network.py'.
                    with torch.no_grad():
                    
                        # Number of shapes in 'fake_point' is equal to the batch size.
                        fake_point = self.G(tree, generator_class_labels)
                    
                    # Evaluate both the ground truth and generated shape using the discriminator.
                    # 'self.D' leads into the 'forward()' function of the generator in 'treegan_network.py'.
                    
                    # Pass the class tensor together with the fake and real shapes to the discriminator.
                    D_real, _ = self.D(point, discriminator_class_labels)
                    D_fake, _ = self.D(fake_point, discriminator_class_labels)

# ++++++++++++++++++++++++++++++++++++++++++++++++++
                    # Compute the gradient penalty loss.
                    if self.args.conditional_gan:
                        gp_loss = self.GP(self.D, point.data, fake_point.data, conditional = True, yreal = discriminator_class_labels)
                    else:
                        gp_loss = self.GP(self.D, point.data, fake_point.data, conditional = False, yreal = discriminator_class_labels)
# ++++++++++++++++++++++++++++++++++++++++++++++++++

                    # Compute discriminator losses.
                    D_realm = D_real.mean()
                    D_fakem = D_fake.mean()
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss

                    # Multiply the loss by the weight before backpropagation.
                    d_loss*=self.w_train
                    d_loss_gp.backward()
                    self.optimizerD.step()

                loss_log['D_loss'].append(d_loss.item())
                epoch_d_loss.append(d_loss.item())
                toc = time.time()
                
# ---------------------- Generator ---------------------- #

                # Reset generator gradients to zero.
                self.G.zero_grad()
                
                # Generate the latent space representation.
                # First dimension of latent space is dependent on batch size.
                z = torch.randn(point.shape[0], 1, latent_space_dim).to(self.args.device)

                # Pass the latent space representation to the generator.
                tree = [z]
                fake_point = self.G(tree, generator_class_labels)
                
                G_fake, _ = self.D(fake_point, discriminator_class_labels)
                G_fakem = G_fake.mean()
                g_loss = -G_fakem

                if self.args.expansion_penality:
                    dist, _, mean_mst_dis = self.expansion(fake_point,self.args.expan_primitive_size,self.args.expan_alpha)
                    expansion = torch.mean(dist)
                    g_loss = -G_fakem + self.args.expan_scalar * expansion

                if self.args.krepul_loss:
                    krepul_loss = self.krepul_loss(fake_point)
                    g_loss = -G_fakem + self.args.krepul_scalar * krepul_loss

                if self.args.knn_loss:
                    knn_loss = self.knn_loss(fake_point)
                    g_loss = -G_fakem + self.args.knn_scalar * knn_loss

                # Multiply the loss by the weight before backpropagation.
                g_loss*=self.w_train
                g_loss.backward()
                self.optimizerG.step()

                loss_log['G_loss'].append(g_loss.item())
                epoch_g_loss.append(g_loss.item())
                tac = time.time()
                # --------------------- Visualization -------------------- #
                verbose = None

                if verbose is not None:
                    print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch + 1, _iter),
                        "[ D_Loss ] ", "{: 7.6f}".format(d_loss),
                        "[ G_Loss ] ", "{: 7.6f}".format(g_loss),
                        "[ Time ] ", "{:4.2f}s".format(time.time()-start_time),
                        "{:4.2f}s".format(toc-tic),
                        "{:4.2f}s".format(tac-toc))

            # ---------------- Epoch everage loss   --------------- #
            d_loss_mean = np.array(epoch_d_loss).mean()
            g_loss_mean = np.array(epoch_g_loss).mean()
            
            # Log each iteration's generator and discriminator losses to the tensorboard summarywriter.
            self.shapeinversion_writer.add_scalar('G_Loss', g_loss_mean, epoch)
            self.shapeinversion_writer.add_scalar('D_Loss', d_loss_mean, epoch)

            print("[Epoch] ", "{:3}".format(epoch + 1),
                "[ D_Loss ] ", "{: 7.6f}".format(d_loss_mean),
                "[ G_Loss ] ", "{: 7.6f}".format(g_loss_mean),
                "[ Time ] ", "{:.2f}s".format(time.time()-epoch_time))
            epoch_time = time.time()

            ### call abstracted eval, which includes FPD
            if self.args.eval_every_n_epoch > 0:
                if (epoch + 1) % self.args.eval_every_n_epoch == 0 :
                    # Ensure that the number of samples taken is smaller than the number of training shapes per epoch.
                    checkpoint_eval(self.G, self.args.device, n_samples = 5000, batch_size = 100, conditional = False, ratio = 'even', FPD_path = self.args.FPD_path, class_choices = self.args.class_choice, latent_space_dim = latent_space_dim)

            # ---------------------- Save checkpoint --------------------- #
            if self.args.save_every_n_epoch > 0:
                if (epoch + 1) % self.args.save_every_n_epoch == 0 and not save_ckpt == None:
                    torch.save({
                            'epoch': epoch + 1,
                            'D_state_dict': self.D.module.state_dict(),
                            'G_state_dict': self.G.module.state_dict(),
                            'D_loss': loss_log['D_loss'],
                            'G_loss': loss_log['G_loss'],
                            'FPD': metric['FPD']
                    }, save_ckpt + str(epoch + 1) + '_' + self.args.class_choice + '.pt')
        
        # Call 'flush()' to write all the recorded tensorboard summarywriter values to disk.
        self.shapeinversion_writer.flush()
        
        # Call 'close()' once the summarywriter is no longer needed.
        self.shapeinversion_writer.close()

if __name__ == '__main__':

    args = Arguments(stage='pretrain').parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    if not osp.isdir('./pretrained_checkpoints'):
        os.mkdir('./pretrained_checkpoints')
        print('pretrain_checkpoints parent directory created.')

    if not osp.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_load if args.ckpt_load is not None else None
    # print(args)

    model = TreeGAN(args)
    model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT)