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

class TreeGAN():
    def __init__(self, args):
        
        print('\n\nINSIDE: pretrain_treegan.py, Class: TreeGAN - __init__')
        self.args = args
        self.args.class_choice = self.args.class_choice.lower()

# --------------------------------------------------------
        # If multiclass pretraining is specified.
        if self.args.class_choice == 'multiclass' and self.args.class_range is not None:
            
            # Convert the one hot encoding list into an array, representing the classes.
            self.classes_chosen = encode_classes(self.args.class_range)
            print('pretrain_treegan.py: __init__ - index of multiclass classes chosen:', self.classes_chosen)
            
            # Set the total number of classes to be used.
            self.total_num_classes = len(self.classes_chosen)
            
            # Create a lookup table using pytorch embedding to represent the number of classes.
            #self.lookup_table = nn.Embedding(self.total_num_classes, 96)
            #print('pretrain_treegan.py - multiclass NN embedding lookup table type:', type(self.lookup_table))
        
        # Otherwise if only using a single class.
        else:
            self.classes_chosen = None
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
        self.G = Generator(features = args.G_FEAT, degrees = args.DEGREE, support = args.support, num_classes = self.total_num_classes, args = self.args).to(args.device)
        self.D = Discriminator(features = args.D_FEAT, num_classes = self.total_num_classes, args = self.args).to(args.device)
        
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
        #self.G = nn.DataParallel(self.G)
        #self.D = nn.DataParallel(self.D)
        
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
                
                # Reshape the class tensor for accomodating the discriminator labels.
                print('pretrain_treegan.py - Class ID tensor size:', class_id.size())
                class_id = torch.reshape(class_id, (4, 1))
                #class_id = class_id.expand(4, 2)
                #print('pretrain_treegan.py - After expanding class ID type:', type(class_id))
                #print('pretrain_treegan.py - After expanding class ID tensor size:', class_id.size())
                print('\nClass ID tensor:', class_id, '\n')
                
                # Perform one hot encoding for the discriminator classes chosen.
                discriminator_class_labels = torch.IntTensor(class_id.shape[0], self.total_num_classes).to(self.args.device)
                
                # Zero all the tensor elements before populating them with the class IDs.
                discriminator_class_labels.zero_()
                print('pretrain_treegan.py - Discriminator class tensor size:', discriminator_class_labels.size())
                print('\nDiscriminator class labels tensor:', discriminator_class_labels, '\n')
                
                # 'scatter' function is used to perform the assignment of one hot encoding values within
                # the class tensor. It sends the elements of the class ID tensor into the specified
                # indices of the 'discriminator_class_labels' tensor.
                
                # CUDA ERROR APPEARS TO BE CAUSED BY THIS LINE USING THE SCATTER FUNCTION.
                discriminator_class_labels.scatter_(1, class_id, 1)
                print('pretrain_treegan.py - Discriminator class tensor size after scattering:', discriminator_class_labels.size())
                print('\nDiscriminator class tensor:', discriminator_class_labels, '\n')
                
                # Convert the resultant tensor into type float.
                discriminator_class_labels.type(torch.cuda.FloatTensor)
                print('\nDResultant discriminator class tensor after converting to float:', discriminator_class_labels, '\n')
                
                # Insert a dimension of size 1 at positional index 1 of the class tensor.
                discriminator_class_labels.unsqueeze_(1)
                print('pretrain_treegan.py - Discriminator class tensor size after unsqueezing:', discriminator_class_labels.size())
                
# -------------------- Discriminator -------------------- #
                tic = time.time()
                
                # Repeat for the number of iterations for the discriminator.
                for d_iter in range(self.args.D_iter):
                    
                    # Reset discriminator gradients to zero.
                    self.D.zero_grad()
                    
                    # CUDA ERROR APPEARS HERE BUT MAY HAVE BEEN CAUSED EARLIER.
                    
                    # Generate the latent space representation.
                    # First dimension of latent space represents the batch size.
                    z = torch.randn(point.shape[0], 1, latent_space_dim).to(self.args.device)
                    
                    # Perform one hot encoding for the generator classes chosen.
                    generator_labels = torch.from_numpy(np.random.randint(0, self.total_num_classes, class_id.shape[0]).reshape(-1, 1)).to(self.args.device)
                    print('pretrain_treegan.py - Generator random labels type:', type(generator_labels))
                    print('pretrain_treegan.py - Generator random labels tensor size:', generator_labels.size())
                    
                    # Create a float tensor for the class IDs.
                    generator_class_labels = torch.FloatTensor(class_id.shape[0], self.total_num_classes).to(self.args.device)
                    
                    # Zero all the tensor elements before populating them with the class IDs.
                    generator_class_labels.zero_()
                    print('pretrain_treegan.py - Generator class tensor size:', generator_class_labels.size())
                    
                    # 'scatter' function is used to perform the assignment of one hot encoding values within
                    # the class tensor. It sends the elements of the class ID tensor into the specified
                    # indices of the 'generator_class_labels' tensor.
                    generator_class_labels.scatter_(1, generator_labels, 1)
                    print('pretrain_treegan.py - Generator class tensor size after scattering:', generator_class_labels.size())
                    
                    # Insert a dimension of size 1 at positional index 1 of the class tensor.
                    generator_class_labels.unsqueeze_(1)
                    print('pretrain_treegan.py - Generator class tensor size after unsqueezing:', generator_class_labels.size())
                    
# --------------------------------------------------------
                    # For multiclass operation, concatenate the latent space tensor with the class ID tensor.
                    #if self.args.class_choice == 'multiclass' and class_id is not None:
                        
                        # Create the embedding layer using the class IDs of the retrieved shapes.
                        #self.embed_layer = self.lookup_table(class_id).to(self.args.device)
                        #print('Discriminator embedding layer shape before unsqueeze:', self.embed_layer.shape)
                        #print('Multiclass discriminator iteration - class embedding layer type:', type(self.embed_layer))
                        
                        # Use 'unsqueeze' operation to insert a dimension of 1 at the first dimension.
                        #self.embed_layer = torch.unsqueeze(self.embed_layer, 1)
                        #print('Discriminator z shape:', z.shape)
                        #print('Multiclass discriminator embedding layer output shape:', self.embed_layer.shape)
                        
                        # Concatenate the tensor representing the class IDs to the latent space representation.
                        #z = torch.cat((z, self.embed_layer), dim = 2)
                        #print('Discriminator - class concatenated with latent space tensor:', z)
                        #print('Multiclass discriminator concatenated tensor shape:', z.shape)
                        # Output tensor shape is (1, 1, 192).
# --------------------------------------------------------
                    
                    # Pass the latent space representation to the generator.
                    tree = [z]
                    print('pretrain_treegan.py - tree[0] shape:', tree[0].size())

                    # Reset the gradients and pass the latent space representation to the generator.
                    # 'self.G' leads into the 'forward()' function of the generator in 'treegan_network.py'.
                    with torch.no_grad():
                    
                        # Number of shapes in 'fake_point' is equal to the batch size.
                        fake_point = self.G(tree, generator_class_labels)

                    # Evaluate both the ground truth and generated shape using the discriminator.
                    # 'self.D' leads into the 'forward()' function of the generator in 'treegan_network.py'.
                    D_real, _ = self.D(point, discriminator_class_labels)
                    D_fake, _ = self.D(fake_point, generator_class_labels)
                    
                    # Compute the gradient penalty loss.
                    gp_loss = self.GP(self.D, point.data, fake_point.data)

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

# --------------------------------------------------------
                # For multiclass operation, concatenate the latent space tensor with the class tensor.
                #if self.args.class_choice == 'multiclass' and class_id is not None:
                        
                    # Create the embedding layer.
                    #self.embed_layer = self.lookup_table(class_id).to(self.args.device)
                    #print('Generator embedding layer shape before unsqueeze:', self.embed_layer.shape)
                    #print('Multiclass generator iteration - class embedding layer type:', type(self.embed_layer))
                    
                    # Use 'unsqueeze' operation to insert a dimension of 1 at the first dimension.
                    #self.embed_layer = torch.unsqueeze(self.embed_layer, 1)
                    #print('Generator z shape:', z.shape)
                    #print('Multiclass generator embedding layer output shape:', self.embed_layer.shape)
                    
                    # Concatenate the tensor representing the classes to the latent space representation.
                    #z = torch.cat((z, self.embed_layer), dim = 2)
                    #print('Generator - class concatenated with latent space tensor:', z)
                    #print('Multiclass generator concatenated tensor shape:', z.shape)
                    # Output tensor shape is (1, 1, 192).
# --------------------------------------------------------

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
