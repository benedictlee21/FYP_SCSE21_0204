import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gcn import TreeGCN

from math import ceil

class Discriminator(nn.Module):
    def __init__(self, batch_size, features,version=0):
        # import pdb; pdb.set_trace()
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        #jz below code got problem, linearity, and final sigmoid,  
        #jz TODO final softmax/sigmoid needed?
        # self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
        #                                  nn.Linear(features[-1], features[-2]),
        #                                  nn.Linear(features[-2], features[-2]),
        #                                  nn.Linear(features[-2], 1))
        
        # follow the r-GAN discriminator, just not very sure if got leaky relu right before sigmoid.
        # jz NOTE below got Sigmoid function
        self.final_layer = nn.Sequential(
                    nn.Linear(features[-1], 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid())

    def forward(self, f):
        
        # feat shape (B,3,2048)
        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
        # import pdb; pdb.set_trace()
        # feat shape (B,dimension,2048) --> out (B,dimension)
        out1 = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        # out (B,1)
        out = self.final_layer(out1) # (B, 1)
        # import pdb; pdb.set_trace()
        return out,out1


class Generator(nn.Module):
    def __init__(self, batch_size, features, degrees, support,version=0,args=None):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()
        
        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            #jz NOTE last layer activation False
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False,args=args))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True,args=args))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, z):
        # import pdb; pdb.set_trace()
        # shape of feat[i] (B,nodes,features)
        # [torch.Size([64, 1, 96]), torch.Size([64, 1, 256]), torch.Size([64, 2, 256]), torch.Size([64, 4, 256]), 
        # torch.Size([64, 8, 128]), torch.Size([64, 16, 128]), torch.Size([64, 32, 128]), torch.Size([64, 2048, 3])]
        tree = [z]
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]
        
        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1] # return a single point cloud (2048,3)

class ConditionalGenerator_v0(nn.Module):
    '''
    conditional GAN, ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
    v0: only cat, do nothing
    '''
    def __init__(self, batch_size, features, degrees, support, n_classes, version=0,args=None):
        super(ConditionalGenerator_v0, self).__init__()
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        self.version = version
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        # self.pointcloud = None
        vertex_num = 1
        self.gcn = nn.Sequential()
        # NOTE: for the first layer, add n_classes
        features[0]+= n_classes
        # NOTE: v1 instead of directly cat and feed into the gcn. feed into a fc first
        if self.version == 1 or self.version == 3:
            self.fc = nn.Sequential(
                nn.Linear(features[0], features[0]),
                nn.LeakyReLU(negative_slope=0.2),
            )
        if self.version == 2 or self.version == 4:
            self.fc = nn.Sequential(
                nn.Linear(features[0], 256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(256, features[0]),
                nn.LeakyReLU(negative_slope=0.2),
            )

        for inx in range(self.layer_num):
            #jz NOTE last layer activation False
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False,args=args))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True,args=args))
            vertex_num = int(vertex_num * degrees[inx])
    def forward(self, tree, labels):
        # shape of feat[i] (B,nodes,features)
        # [torch.Size([64, 1, 96]), torch.Size([64, 1, 256]), torch.Size([64, 2, 256]), torch.Size([64, 4, 256]), 
        # torch.Size([64, 8, 128]), torch.Size([64, 16, 128]), torch.Size([64, 32, 128]), torch.Size([64, 2048, 3])]
        # import pdb; pdb.set_trace()
        if self.version == 0:
            tree[0] = torch.cat((tree[0],labels),-1)
        else:
            tree[0] = self.fc(torch.cat((tree[0],labels),-1))
        
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]
       
        return self.pointcloud

class ConditionalDiscriminator_v0(nn.Module):
    def __init__(self, batch_size, features, n_classes,version=0):
        super(ConditionalDiscriminator_v0, self).__init__()
        # import pdb; pdb.set_trace()
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        self.version = version

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        #jz below code got problem, linearity, and final sigmoid,  
        #jz TODO final softmax/sigmoid needed?
        # self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
        #                                  nn.Linear(features[-1], features[-2]),
        #                                  nn.Linear(features[-2], features[-2]),
        #                                  nn.Linear(features[-2], 1))
        
        # follow the r-GAN discriminator, just not very sure if got leaky relu right before sigmoid.
        # jz NOTE below got Sigmoid function
        feat_dim = features[-1] + n_classes
        # NOTE: v1 instead of directly cat and feed into the gcn. feed into a fc first
        # TODO version 3 and version 4 for D has never been touched.
        if self.version == 3:
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LeakyReLU(negative_slope=0.2),
            )
        if self.version == 4:
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, 1024),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(1024, feat_dim),
                nn.LeakyReLU(negative_slope=0.2),
            )

        self.final_layer = nn.Sequential(
                    nn.Linear(feat_dim, 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid())
    
    def forward(self, f, y):
        # y shape (B, n_classes)        
        # feat shape (B,3,2048)
        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
        # import pdb; pdb.set_trace()
        # feat shape (B,dimension,2048) --> out (B,dimension)
        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        
        # NOTE cat here
        if self.version == 3 or self.version == 4:
            out = torch.cat((out,y.squeeze(1)),-1)
            out1 = self.fc(out)
        else:
            out1 = torch.cat((out,y.squeeze(1)),-1)

        # out (B,1)
        out = self.final_layer(out1) # (B, 1)
        # import pdb; pdb.set_trace()
        return out, out1