import torch
import torch.nn as nn
import torch.nn.functional as func
# from layers.gcn import TreeGCN
from model.gcn import TreeGCN
from math import ceil

class Discriminator(nn.Module):
    def __init__(self, features, classes_chosen = None, version=0):
    
        # Get the number of layers for the discriminator network.
        self.layer_num = len(features)-1
        
        # For class inheritance.
        super(Discriminator, self).__init__()
        
        # Create a list to hold the submodules for fully connected layers.
        self.fc_layers = nn.ModuleList([])

        if classes_chosen is not None:
            print('treegan_network.py: Discriminator initialization - classes chosen:', classes_chosen)
            print('Discriminator operating in multiclass conditional mode.')
                    
            # Need to add the number of multiclass classes to the last value of the discriminator features list.
            features[-1] += len(classes_chosen)

        # Append the discriminator features to each layer of the discriminator network.
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        # Define the activation function.
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        # Define the final layer of the discriminator network.
        self.final_layer = nn.Sequential(
                    nn.Linear(features[-1], 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid())

    # Pretraining does not pass one hot encoded classes to discriminator forward.
    # Additional dimensions were already added to latent space in 'pretrain_treegan.py'.
    def forward(self, tree, classes_chosen = None, device = None):

        feat = tree.transpose(1,2)
        vertex_num = feat.size(2)

        # Pass the point cloud through each layer of the discriminator network.
        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
            
        # Output pooling layer.
        out = func.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        
        if classes_chosen is not None:
            print('treegan_network.py: Discriminator forward - classes chosen:', classes_chosen)
            
            # Convert the one hot encoded array into a tensor for concatenation.
            classes_chosen = torch.from_numpy(classes_chosen)
            
            # Concatenate the multiclass labels with the completed shape.
            # Need to ensure that both tensors are on the same device.
            out = torch.cat((out, classes_chosen.to(device).squeeze(1)), -1)
        
        # Apply the final layer of the network.
        out1 = self.final_layer(out) # (B, 1)
        return out1, out

class Generator(nn.Module):
    def __init__(self, features, degrees, support, classes_chosen = None, args = None):
    
        # Get the number of layers for the generator network.
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        
        # For class instantiation.
        super(Generator, self).__init__()
        vertex_num = 1
        
        # Create a sequential container to hold submodules for the generator network.
        self.gcn = nn.Sequential()

        if classes_chosen is not None:
            print('treegan_network.py: Generator initialization - classes chosen:', classes_chosen)
            print('Generator operating in conditional multiclass mode.')
            
            # First value in generator features list corresponds to the number of dimensions of the latent space.
            # Need to add the number of multiclass classes to this value.
            features[0] += len(classes_chosen)

        # Define the generator network.
        for inx in range(self.layer_num):
            # NOTE last layer activation False
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees,
                                            support=support, node=vertex_num, upsample=True, activation=False,args=args))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees,
                                            support=support, node=vertex_num, upsample=True, activation=True,args=args))
            vertex_num = int(vertex_num * degrees[inx])

    # Pretraining does not pass one hot encoded classes to generator forward.
    # Additional dimensions were already added to latent space in 'pretrain_treegan.py'.
    def forward(self, tree, classes_chosen = None, device = None):
            
        # Obtain all the generated shapes from the result of the graph convolutional network.
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
