import torch
import torch.nn as nn
import torch.nn.functional as func
# from layers.gcn import TreeGCN
from model.gcn import TreeGCN
from math import ceil

class Discriminator(nn.Module):
    def __init__(self, features, num_classes, args = None):
    
        self.args = args
        # Get the number of layers for the discriminator network.
        self.layer_num = len(features)-1
        
        # For class inheritance.
        super(Discriminator, self).__init__()
        
        # Create a list to hold the submodules for fully connected layers.
        self.fc_layers = nn.ModuleList([])

        # Append the discriminator features to each layer of the discriminator network.
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        # Define the activation function.
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
# --------------------------------------------------------
        # Add the required number of dimensions to the input layer of the network for multiclass.
        if self.args.class_choice == 'multiclass' and self.args.conditional_gan:
            print('treegan_network.py - Conditional GAN discriminator number of classes chosen:', num_classes)
            features[-1] += num_classes
            
            # For multiclass shape completion in diversity mode using conditional GAN,
            # only one class may be completed at a time, hence the number of latent space
            # dimensions must be manually added to match the original multiclass model's dimensions.
            if self.args.split == 'test':
                if self.args.inversion_mode == 'multiclass' and num_classes == 1:
                    features[-1] += 1
            
            print('Discriminator features[-1]:', features[-1])
            
            # Create additional network layers for multiclass using conditonal GAN.
            self.fully_connected_V1 = nn.Sequential(
                nn.Linear(features[-1], features[-1]),
                nn.LeakyReLU(negative_slope = 0.2)
            )
            
            self.fully_connected_V2 = nn.Sequential(
                nn.Linear(features[-1], 1024),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.Linear(1024, features[-1]),
                nn.LeakyReLU(negative_slope = 0.2)
            )
# --------------------------------------------------------
        
        # Define the final layer of the discriminator network.
        self.final_layer = nn.Sequential(
                    nn.Linear(features[-1], 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
        )

    # Pretraining does not use the forward propagation function.
    def forward(self, tree, class_labels = None, device = None):

        feat = tree.transpose(1,2)
        vertex_num = feat.size(2)

        # Pass the point cloud through each layer of the discriminator network.
        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
            
        # Output pooling layer.
        out = func.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        
# --------------------------------------------------------
        # For multiclass operation, concatenate the discriminator output with the number of classes.
        if self.args.class_choice == 'multiclass' and self.args.conditional_gan and class_labels is not None:
            #print('treegan_network.py - Concatenating discriminator output with class label.')
            out = torch.cat((out, class_labels.squeeze(1)), -1)
            
            # Alternatively, apply the additional fully connected layers from earlier, V1 or V2.
            out = self.fully_connected_V1(out)
            #out = self.fully_connected_V2(out)
# --------------------------------------------------------
        
        # Apply the final layer of the network.
        out1 = self.final_layer(out) # (B, 1)
        return out1, out

class Generator(nn.Module):
    def __init__(self, features, degrees, support, num_classes, args = None):
        
        self.args = args
        # Get the number of layers for the generator network.
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        
        # For class instantiation.
        super(Generator, self).__init__()
        vertex_num = 1
        
        # Create a sequential container to hold submodules for the generator network.
        self.gcn = nn.Sequential()
        
# --------------------------------------------------------
        # Add the required number of dimensions to the input layer of the network for multiclass.
        if self.args.class_choice == 'multiclass' and self.args.conditional_gan:
            print('treegan_network.py - Conditional GAN generator number of classes chosen:', num_classes)
            features[0] += num_classes
            
            # For multiclass shape completion in diversity mode using conditional GAN,
            # only one class may be completed at a time, hence the number of latent space
            # dimensions must be manually added to match the original multiclass model's dimensions.
            if self.args.split == 'test':
                if self.args.inversion_mode == 'multiclass' and num_classes == 1:
                    features[0] += 1
            
            print('Generator features[0]:', features[0])
        
            # Create additional network layers for multiclass using conditional GAN.
            self.fully_connected_V1 = nn.Sequential(
                nn.Linear(features[0], features[0]),
                nn.LeakyReLU(negative_slope = 0.2)
            )
            
            self.fully_connected_V2 = nn.Sequential(
                nn.Linear(features[0], 256),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.Linear(256, features[0]),
                nn.LeakyReLU(negative_slope = 0.2)
            )
# --------------------------------------------------------

        # Define each layer of the generator network.
        for inx in range(self.layer_num):
            
            # The last layer's activation is false.
            if inx == self.layer_num - 1:
                self.gcn.add_module('TreeGCN_' + str(inx),
                    TreeGCN(inx, features, degrees,
                    support = support, node = vertex_num,upsample = True, activation = False, args = args))
            else:
                self.gcn.add_module('TreeGCN_' + str(inx),
                    TreeGCN(inx, features, degrees,
                    support = support, node = vertex_num, upsample = True, activation = True, args = args))
            vertex_num = int(vertex_num * degrees[inx])

    # Pretraining does not use the forward propagation function.
    def forward(self, tree, class_labels = None, device = None):

# --------------------------------------------------------
        # For multiclass operation, concatenate the latent space with the class labels.
        if self.args.class_choice == 'multiclass' and self.args.conditional_gan and class_labels is not None:
            #print('treegan_network.py - Concatenating generator output with class label.')
            
            # Uncomment the line below for running using the original conditional GAN.
            #tree[0] = torch.cat((tree[0], class_labels), -1)
            #print('Concatenated latent space size:', tree[0].size())
        
            # Alternatively, apply the additional fully connected layers from earlier, V1 or V2.
            tree[0] = self.fully_connected_V1(torch.cat((tree[0], class_labels), -1))
            #tree[0] = self.fully_connected_V2(torch.cat((tree[0], class_labels), -1))
# --------------------------------------------------------
        
        # Pass the network features to the graph convolutional network.
        # 'self.gcn' leads to the 'forward' function of the 'TreeGAN' class in 'gcn.py'.
        feat = self.gcn(tree)

        # Use only the last shape of the result as the output.
        self.pointcloud = feat[-1]
        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1] # return a single point cloud (2048,3)

    def get_params(self,index):

        if index < 7:
            for param in self.gcn[index].parameters():
                yield param
        else:
            raise ValueError('Index out of range')
