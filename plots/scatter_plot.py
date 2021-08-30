# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

### ref: https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
#     print(m,zlow,zhigh)
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, marker=m)

# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/uni_loss_v0_1975/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/branch_v1_1220/30.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/branch_v2_1040/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/branch_v3_1310/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/branch_v4_1200/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/cus_los0/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/cus_los28/20.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/knn5/42.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/knn6/45.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul1/12.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul2/19.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul3/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul4/10.txt'
pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/knn/knn6/20.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul6/30.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul7/30.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul8/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul9/18.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul10/9.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul11/14.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/repul12/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/expan1/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/expan2/20.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/expan3/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/expan4/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/expan5/10.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/expan6/20.txt'
# pcd_pathname = '/Users/zhangjunzhe/Downloads/pointsets/expan7/20.txt'
pcd = np.loadtxt(pcd_pathname,delimiter=';').astype(np.float32)
# for l in pcd:
    # print(l)

# .astype(np.float32)
# print(pcd.shape)
# xs = pcd[:,0]
# ys = pcd[:,2]
# zs = pcd[:,1]



### show all PCD with 32 different colors
mode = 1
n_groups = 32
n_per_group = int(2048/n_groups)
cnt = 0
x_lim = (-0.2,-0.0)
y_lim = (-0.2,0.2)
z_lim = (-0.38,-0.0)
if mode == 1:
    colors = cm.rainbow(np.linspace(0,1,n_groups))
    for i,c in enumerate(colors):
        # if i > 5:
        #     continue
        xs = pcd[i*n_per_group:(i+1)*n_per_group,0]
        ys = pcd[i*n_per_group:(i+1)*n_per_group,2]
        zs = pcd[i*n_per_group:(i+1)*n_per_group,1]
        ax.scatter(xs, ys, zs,s=5,color=c,label=str(i))
        cnt+= xs.shape[0]
    
elif mode == 2:
    # include = range(0,5)
    include = [10, 30, 20, 5, 0]
    colors = cm.rainbow(np.linspace(0,1,len(include)))
    for idx, group in enumerate(include):
        xs = pcd[group*n_per_group:(group+1)*n_per_group,0]
        ys = pcd[group*n_per_group:(group+1)*n_per_group,2]
        zs = pcd[group*n_per_group:(group+1)*n_per_group,1]
        ax.scatter(xs, ys, zs,s=5,color=colors[idx],label=str(group))
        cnt+= xs.shape[0]
elif mode == 3: # point certain area only
    x1, x2 = x_lim
    y1, y2 = y_lim
    z1, z2 = z_lim
    colors = cm.rainbow(np.linspace(0,1,n_groups))
    colors_painted = []
    PCD_dict = {i:np.empty((0,0)) for i in range(n_groups)}
    for i in range(2048):
        xs = pcd[i,0]
        ys = pcd[i,2]
        zs = pcd[i,1]
        if x1 < xs < x2 and y1 < ys < y2 and z1 < zs < z2:
            if PCD_dict[int(i/n_per_group)].shape[0] > 0:
            # import pdb; pdb.set_trace()
                PCD_dict[int(i/n_per_group)] = np.concatenate((PCD_dict[int(i/n_per_group)],pcd[i].reshape(1,-1)))
            else:
                PCD_dict[int(i/n_per_group)] = pcd[i].reshape(1,-1)
            # print(int(i/n_per_group),PCD_dict[int(i/n_per_group)].shape)
            # c = colors[int(i/n_per_group)]
            # colors_painted.append(i%n_groups)
            # if i%n_groups not in colors_painted:
            # ax.scatter(xs, ys, zs,s=5,color=c,label=str(int(i/n_per_group)))
    for k, value in PCD_dict.items():
        if value.shape[0] > 0:
            # import pdb; pdb.set_trace()
            xs = value[:,0]
            ys = value[:,2]
            zs = value[:,1]
            ax.scatter(xs, ys, zs,s=5,color=colors[k],label=str(k))

print (cnt)
# if include is not None:
#     xs = np.empty(0)
#     ys = np.empty(0)
#     zs = np.empty(0)
#     for i in include:
#         xs = np.concatenate((pcd[i*n_groups:(i+1)*n_groups,0],xs))
#         ys = np.concatenate((pcd[i*n_groups:(i+1)*n_groups,2],ys))
#         zs = np.concatenate((pcd[i*n_groups:(i+1)*n_groups,1],zs))
# print('shape of xs',xs.shape)
# ax.scatter(xs, ys, zs,s=1)

## just to disapprove how the 2048 points is ordered.
# for i in range(2048):
#     c = colors[i%n_groups]
#     xs = pcd[i,0]
#     ys = pcd[i,2]
#     zs = pcd[i,1]
#     ax.scatter(xs, ys, zs,s=1,color=c,label=str(i))

# ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
# bbox_to_anchor=(0.5, 0., 0.5, 0.5)
ax.legend(loc='best',ncol=2,bbox_to_anchor=(1.2,1),markerscale=5)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# # standard
ax.set_xlim([-0.5,0.5])
ax.set_ylim([-0.5,0.5])
ax.set_zlim([-0.5,0.5])

# to show cluttered area only
# ax.set_xlim([-0.5,0.0])
# ax.set_ylim([-0.2,0.2])
# ax.set_zlim([-0.2,0.0])

plt.subplots_adjust(right=0.8)
plt.show()