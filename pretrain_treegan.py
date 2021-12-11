import torch
import torch.nn as nn
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
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
import tensorflow

class TreeGAN():
    def __init__(self, args):
        self.args = args

        # If multiclass pretraining is specified.
        if args.class_range is not None:
            
            # Convert the one hot encoding list into an array, representing the classes.
            self.classes_chosen = encode_classes(args.class_range)
            print('\nchair, table, couch, cabinet, lamp, car, plane, watercraft')
            print('pretrain_treegan.py: __init__ classes chosen:', self.classes_chosen)
        
        # Otherwise if only using a single class.
        else:
            self.classes_chosen = None

        # Load the dataset. Reduce the number of workers here if the data loading process freezes.
        # Original number of workers is 16.
        self.data = CRNShapeNet(args, self.classes_chosen)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        print("Training Dataset : {} prepared.".format(len(self.data)))

        # Define the generator and discriminator models.
        # Pass in the chosen classes if multiclass is specified.
        self.G = Generator(features = args.G_FEAT, degrees = args.DEGREE, support = args.support, classes_chosen = self.classes_chosen, args=self.args).to(args.device)
        self.D = Discriminator(features = args.D_FEAT, classes_chosen = self.classes_chosen).to(args.device)
        
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

        for epoch in range(epoch_log, self.args.epochs):
            print('Starting epoch:', epoch + 1)
            epoch_g_loss = []
            epoch_d_loss = []
            epoch_time = time.time()
            self.w_train = self.w_train_ls[min(3,int(epoch/500))]

            for _iter, data in enumerate(self.dataLoader):
                # Start Time
                start_time = time.time()
                point, _, _ = data
                point = point.to(self.args.device)

# -------------------- Discriminator -------------------- #

                tic = time.time()
                for d_iter in range(self.args.D_iter):
                    
                    # Reset discriminator gradients to zero.
                    self.D.zero_grad()
                    
                    # For multiclass, use an embedding layer to create a vector with the same dimensions
                    # as the latent space representation to indicate the class and combine it with the latent space.
                    if self.classes_chosen is not None:
                    
                        # Alternative using torch.nn.Embedding.
                        # Arguments: <number of embeddings>, <embedding dimensions>, <padding index>
                        classes_embedding = nn.Embedding(8, 96, padding_idx = 0)
                        
                        # Generate the latent space representation for single class.
                        z = torch.randn(point.shape[0], 1, latent_space_dim).to(self.args.device)
                        
                        # Output the corresponding word embeddings.
                        z = classes_embedding(z)
                        
                        print('pretrain_treegan.py - Discriminator')
                        print('Pytorch embedding layer type:', type(z))
                        print('Pytorch embedding layer output:', z)
                        #print('Pytorch embedding layer shape:', z.shape)
                        
                        # -------------------------------------------------------------------------------
                    
                        # Prepare the latent space for concatenation with the class vector.
                        latent_space = Input(shape = (latent_space_dim, ))
                        
                        # Define the class labels using an instantiated tensor.
                        class_labels = Input(shape = (1, ), dtype = 'int32')
                    
                        # Input dimensions should be the number of classes.
                        # Output dimensions should be the same as the latent space representation.
                        classes_embedding = Embedding(input_dim = len(self.classes_chosen), output_dim = latent_space_dim, input_length = 1)(class_labels)
                        
                        # Flatten the tensor representing class labels into a single dimension.
                        #classes_embedding = Flatten()(classes_embedding)
                        
                        print('pretrain_treegan.py - Discriminator')
                        print('Embedding layer type:', type(classes_embedding))
                        print('Embedding layer output:', classes_embedding)
                        #print('Embedding layer shape:', classes_embedding.shape)
                        
                        # Combine the latent space with the class embedding.
                        z = Multiply()([latent_space, classes_embedding])
                        
                        # Reshape the tensor into the required input dimensions.
                        # First dimension of the output tensor is 'None', indicating that it is an unspecified
                        # dimension for use with multiclass capabilities.
                        #torch.reshape(z, (1, 1, 96))
                        tensorflow.reshape(z, [1, 1, 96])
                        
                        print('Concatenated latent space type:', type(z))
                        print('Concatenated latent space shape:', z.shape)
                        
                        # Resultant tensor is of shape (None, 1, 96).
                        # Need to convert it to (1, 1, 96).
                        
                    else:
                        # Generate the latent space representation for single class.
                        z = torch.randn(point.shape[0], 1, latent_space_dim).to(self.args.device)
                        
                    # Store the latent space in a list.
                    tree = [z]

                    # One hot encoded array for multiclass classes chosen is not
                    # passed into the forward function of the generator during pretraining.
                    with torch.no_grad():
                        fake_point = self.G(tree)

                    D_real, _ = self.D(point)
                    D_fake, _ = self.D(fake_point)
                    gp_loss = self.GP(self.D, point.data, fake_point.data)

                    # compute D loss
                    D_realm = D_real.mean()
                    D_fakem = D_fake.mean()
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    
                    # times weight before backward
                    d_loss*=self.w_train
                    d_loss_gp.backward()
                    self.optimizerD.step()

                loss_log['D_loss'].append(d_loss.item())
                epoch_d_loss.append(d_loss.item())
                toc = time.time()
                
# ---------------------- Generator ---------------------- #

                # Reset generator gradients to zero.
                self.G.zero_grad()
                
                # For multiclass, use an embedding layer to create a vector with the same dimensions
                # as the latent space representation to indicate the class and combine it with the latent space.
                if self.classes_chosen is not None:
                
                    # Alternative using torch.nn.Embedding.
                    # Arguments: <number of embeddings>, <embedding dimensions>, <padding index>
                    classes_embedding = nn.Embedding(8, 96, padding_idx = 0)
                        
                    # Generate the latent space representation for single class.
                    z = torch.randn(point.shape[0], 1, latent_space_dim).to(self.args.device)
                        
                    # Output the corresponding word embeddings.
                    z = classes_embedding(z)
                        
                    print('pretrain_treegan.py - Generator')
                    print('Pytorch embedding layer type:', type(z))
                    print('Pytorch embedding layer output:', z)
                    #print('Pytorch embedding layer shape:', z.shape)
                    
                    # -------------------------------------------------------------------------------
                    
                    # Prepare the latent space for concatenation with the class vector.
                    latent_space = Input(shape = (latent_space_dim, ))
                        
                    # Define the class labels using an instantiated tensor.
                    class_labels = Input(shape = (1, ), dtype = 'int32')
                    
                    # Input dimensions should be the number of classes.
                    # Output dimensions should be the same as the latent space representation.
                    classes_embedding = Embedding(input_dim = len(self.classes_chosen), output_dim = latent_space_dim, input_length = 1)(class_labels)
                        
                    # Flatten the tensor representing class labels into a single dimension.
                    #classes_embedding = Flatten()(classes_embedding)
                        
                    print('pretrain_treegan.py - Generator')
                    print('Embedding layer type:', type(classes_embedding))
                    print('Embedding layer output:', classes_embedding)
                    #print('Embedding layer shape:', classes_embedding.shape)
                        
                    # Combine the latent space with the class embedding.
                    z = Multiply()([latent_space, classes_embedding])
                    
                    # Reshape the tensor into the required input dimensions.
                    # First dimension of the output tensor is 'None', indicating that it is an unspecified
                    # dimension for use with multiclass capabilities.
                    #torch.reshape(z, (1, 1, 96))
                    tensorflow.reshape(z, [1, 1, 96])
                    
                    print('Concatenated latent space type:', type(z))
                    print('Concatenated latent space shape:', z.shape)
                    
                    # Resultant tensor is of shape (None, 1, 96).
                    # Need to convert it to (1, 1, 96).
                        
                else:
                    # Generate the latent space representation for single class.
                    z = torch.randn(point.shape[0], 1, latent_space_dim).to(self.args.device)

                # Store the latent space in a list.
                tree = [z]
                fake_point = self.G(tree)
                
                # One hot encoded array for multiclass classes chosen is not
                # passed into the forward function of the generator during pretraining.
                G_fake, _ = self.D(fake_point)
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
