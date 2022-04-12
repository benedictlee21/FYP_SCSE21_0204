import os
import torch
import numpy as np
from external.ChamferDistancePytorch.chamfer_python import distChamfer
import glob
from metrics import *
from loss import *
import h5py
from utils.common_utils import *
from arguments import Arguments

def compute_ucd(partial_ls, output_ls):
    """ 
    input two lists (small lists)
    return a single mean
    """
    if isinstance(partial_ls[0],np.ndarray):
        partial_ls = [torch.from_numpy(itm) for itm in partial_ls]
        output_ls = [torch.from_numpy(itm) for itm in output_ls]

    if len(partial_ls) < 100:
        partial = torch.stack(partial_ls).cuda()
        output = torch.stack(output_ls).cuda()
        dist1, dist2 , _, _ = distChamfer(partial, output)
        cd_loss = dist1.mean()*10000
        cd_ls = (dist1.mean(1)*10000).cpu().numpy().tolist()
        return cd_loss.item(), cd_ls

    else:
        batch_size = 50
        n_samples = len(partial_ls)
        n_batches = int(n_samples/batch_size) + min(1, n_samples%batch_size)
        cd_ls = []
        for i in range(n_batches):
            # if i*batch_size
            # print(n_samples, i, i*batch_size)
            partial = torch.stack(partial_ls[i*batch_size:min(n_samples,i*batch_size+batch_size)]).cuda()
            output = torch.stack(output_ls[i*batch_size:min(n_samples,i*batch_size+batch_size)]).cuda()
            dist1, dist2 , _, _ = distChamfer(partial, output)
            cd_loss = dist1.mean(1)*10000
            cd_ls.append(cd_loss)
        cd = torch.cat(cd_ls).mean().item()
        cd_ls = torch.cat(cd_ls).cpu().numpy().tolist()
        return cd, cd_ls

def compute_uhd(partial_ls, output_ls):
    """
    input two lists (small lists)
    return a single mean
    """
    if isinstance(partial_ls[0],np.ndarray):
        partial_ls = [torch.from_numpy(itm) for itm in partial_ls]
        output_ls = [torch.from_numpy(itm) for itm in output_ls]
    partial = torch.stack(partial_ls).cuda()
    output = torch.stack(output_ls).cuda()
    uhd = DirectedHausdorff()
    udh_loss = uhd(partial, output)
    return udh_loss.item()

### FUNCTION IMMEDIATELY BELOW FOR SHAPEINVERSION DIVERSITY RESULTS ONLY

def eval_completion_without_gt(input_dir):
    # Retrive the partial and completed shapes from their directory.
    pathnames = glob.glob(input_dir + '*')

    # Lists to store the partial and completed shapes.
    input_partials = []
    output_shapes = []

    # For each file in the directory, check if it is a partial or completed shape.
    # Store the filename of that file into their respective list.
    for filepath in pathnames:
        if 'X_Partial_Shape.txt' in filepath:
            input_partials.append(filepath)
        elif 'Completed_Shape.txt' in filepath:
            output_shapes.append(filepath)

    # Sort the shapes alphabetically.
    sorted_inputs = sorted(input_partials)
    sorted_outputs = sorted(output_shapes)
    ours_input = []
    ours_output = []

    # Read in the point cloud shape coordinates from each file and convert it into a numpy array.
    for i in sorted_inputs:
         input_numpy = np.loadtxt(i, delimiter = ';').astype(np.float32)
         ours_input.append(input_numpy)

    for j in sorted_outputs:
         output_numpy = np.loadtxt(j, delimiter = ';').astype(np.float32)
         ours_output.append(output_numpy)

    # Compute the unidirectional chamfer distance.
    cd, cd_ls = compute_ucd(ours_input, ours_output)
    
    # Compute the unidirectional hausdorff distance.
    uhd = compute_uhd(ours_input, ours_output)
    
    print('Current directory:', input_dir)
    print('UCD: ', cd)
    print('UHD: ', uhd)

# This file can run as a standalone program.
if __name__ == '__main__':

    # Read in the required arguments.
    args = Arguments(stage='eval_completion').parser().parse_args()
    input_dir = args.saved_results_path
    
    # No ground truth is used for multiclass evaluations.
    eval_completion_without_gt(input_dir)
