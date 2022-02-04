import torch

def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    P = rx.t() + ry - 2 * zz
    return P

def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()

# Modified chamfer distance function that uses K-mask.
def distChamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -chamfer distance of point cloud a
    -chamfer distance of point cloud b
    -minimum argument of point cloud a
    -minimum argument of point cloud b
    Works for pointcloud of any dimension
    """
    # Convert point cloud points into data type 'double'.
    x, y = a.double(), b.double()
    
    # Obtain the batch size, number of points and number of dimensions for each shape.
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()
    
    # Raise each element in both point clouds by a power of 2 and return the sum of all elements in each one respectively.
    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    
    # Perform batch by batch matrix multiplication of all matrices stored in point clouds 'X' and transpose of point cloud 'Y'.
    zz = torch.bmm(x, y.transpose(2, 1))
    
    # Add a new dimension to the tensor along dimension 1 and expand it to a larger size.
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    
    # Find the chamfer distance between the two point clouds.
    P = rx.transpose(2, 1) + ry - 2 * zz
    
    # Difference between original chamfer distance and K-mask chamfer distance is in this line.
    # Returns the minimum value of the resulting tensor elements.
    return torch.min(P, 2)[0].float(), torch.min(P, 1)[0].float(), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()

# Original distance chamfer implementation without using K-mask.
def distChamfer_raw(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P
