import torch
import torch.nn as nn
import torch.nn.functional as func
# from layers.gcn import TreeGCN
from model.gcn import TreeGCN
from math import ceil

class Discriminator(nn.Module):
    def __init__(self, features, total_num_classes, args = None):
    
        self.args = args
        # Get the number of layers for the discriminator network.
        self.layer_num = len(features)-1
        
        # For class inheritance.
        super(Discriminator, self).__init__()
        
        # Create a list to hold the submodules for fully connected layers.
        self.fc_layers = nn.ModuleList([])

        # Append the discriminator features to each layer of the discriminator network.
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx + 1], kernel_size = 1, stride = 1))

        # Define the activation function.
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        
# --------------------------------------------------------
        # Add the required number of dimensions to the input layer of the network for multiclass.
        if self.args.class_choice == 'multiclass' and self.args.conditional_gan:
            features[-1] += total_num_classes
            #print('Discriminator features[-1]:', features[-1])
# --------------------------------------------------------
        
        # Define the final layer of the discriminator network.
        self.final_layer = nn.Sequential(
                    nn.Linear(features[-1], 128),
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
        )

    # Pretraining does not use the forward propagation function.
    def forward(self, tree, class_labels = None):

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
            
            # Remove dimension 1 of the class labels tensor if its size is 1.
            out = torch.cat((out, class_labels.squeeze(1)), -1)
# --------------------------------------------------------
        
        # Apply the final layer of the network.
        out1 = self.final_layer(out) # (B, 1)
        return out1, out

class Generator(nn.Module):
    def __init__(self, features, degrees, support, total_num_classes, args = None):
        
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
            features[0] += total_num_classes
            #print('Generator features[0]:', features[0])
# --------------------------------------------------------

        # Define each layer of the generator network.
        for inx in range(self.layer_num):
            
            # The last layer's activation is false.
            if inx == self.layer_num - 1:
                self.gcn.add_module('TreeGCN_' + str(inx),
                    TreeGCN(inx, features, degrees,
                    support = support, node = vertex_num, upsample = True, activation = False, args = args))
            else:
                self.gcn.add_module('TreeGCN_' + str(inx),
                    TreeGCN(inx, features, degrees,
                    support = support, node = vertex_num, upsample = True, activation = True, args = args))
            vertex_num = int(vertex_num * degrees[inx])

    # Pretraining does not use the forward propagation function.
    def forward(self, tree, class_labels = None):

# --------------------------------------------------------
        # For multiclass operation, concatenate the latent space with the class labels.
        if self.args.class_choice == 'multiclass' and self.args.conditional_gan and class_labels is not None:
            #print('treegan_network.py - Concatenating generator output with class label.')
            
            # Uncomment the line below for running using the original conditional GAN.
            # Shape of class labels is (<batch size>, 1, <total number of classes>).
            #print('Class labels shape:', class_labels.shape)
            tree[0] = torch.cat((tree[0], class_labels), -1)
            #print('Concatenated latent space size:', tree[0].size())
# --------------------------------------------------------
        
        # Pass the network features to the graph convolutional network.
        # 'self.gcn' leads to the 'forward' function of the 'TreeGAN' class in 'gcn.py'.
        # Shape of tree[0] is (<batch size>, 1, <latent space dims + number of classes>).
        #print('Tree[0] shape:', tree[0].shape)
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
