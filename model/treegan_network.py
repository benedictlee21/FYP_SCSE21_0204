import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.gcn import TreeGCN
from model.gcn import TreeGCN
from math import ceil

class Discriminator(nn.Module):
    def __init__(self, features, classes_chosen = None, version=0):

        if classes_chosen is not None:
            print('treegan_network.py: Discriminator initialization - classes chosen:', classes_chosen)
            print('Discriminator operating in multiclass conditional mode.')
                    
            # Need to add the number of multiclass classes to the last value of the discriminator features list.
            features[-1] += len(classes_chosen)

        self.layer_num = len(features)-1
        super(Discriminator, self).__init__()
        self.fc_layers = nn.ModuleList([])

        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.final_layer = nn.Sequential(
                    nn.Linear(features[-1], 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid())

    def forward(self, f, classes_chosen = None):

        if classes_chosen is not None:
            print('treegan_network.py: Discriminator forward - classes chosen:', classes_chosen)

        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        
        if classes_chosen is not None:
            # Concatenate the multiclass labels with the completed shape.
            out = torch.cat((out, classes_chosen.squeeze(1)), -1)
        
        out1 = self.final_layer(out) # (B, 1)
        return out1, out

class Generator(nn.Module):
    def __init__(self, features, degrees, support, classes_chosen = None, args = None):

        if classes_chosen is not None:
            print('treegan_network.py: Generator initialization - classes chosen:', classes_chosen)
            print('Generator operating in conditional multiclass mode.')
            
            # First value in generator features list corresponds to the number of dimensions of the latent space.
            # Need to add the number of multiclass classes to this value.
            features[0] += len(classes_chosen)

        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()

        vertex_num = 1
        self.gcn = nn.Sequential()
        
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

    def forward(self, tree, classes_chosen = None):

        if classes_chosen is not None:
            print('treegan_network.py: Generator forward - classes chosen:', classes_chosen)
            
            # Concatenate the multiclass labels with the generated latent space.
            tree[0] = torch.cat((tree[0], classes_chosen), -1)
            
            # Obtain all the generated shapes from the result of the graph convolutional network.
            feat = self.gcn(tree)
        else:
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
