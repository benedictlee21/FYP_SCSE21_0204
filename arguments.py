import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        # Dataset arguments
        self._parser.add_argument('--dataset',type=str,default='BenchmarkDataset')
        self._parser.add_argument('--dataset_path', type=str, default='/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0', help='Dataset file path.')
        self._parser.add_argument('--class_choice', type=str, default='Chair', help='Select one class to generate. [Airplane, Chair, ...] (default:all_class)')
        #jz TODO batch size default 20
        self._parser.add_argument('--batch_size', type=int, default=64, help='Integer value for batch size.')
        self._parser.add_argument('--point_num', type=int, default=2048, help='Integer value for number of points.')

        # Training arguments
        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        #jz TODO epoch default 2000
        self._parser.add_argument('--epochs', type=int, default=2, help='Integer value for epochs.')
        self._parser.add_argument('--pretrain_D_lr', type=float, default=1e-4, help='Float value for learning rate.')
        self._parser.add_argument('--pretrain_G_lr', type=float, default=1e-4, help='Float value for learning rate.')
        #jz NOTE if do grid search, need to specify the ckpt_path and ckpt_save.
        self._parser.add_argument('--ckpt_path', type=str, default='./model/checkpoints2/', help='Checkpoint path.')
        self._parser.add_argument('--ckpt_save', type=str, default='tree_ckpt_', help='Checkpoint name to save.')
        self._parser.add_argument('--ckpt_load', type=str, help='Checkpoint name to load. (default:None)')
        self._parser.add_argument('--result_path', type=str, default='./model/generated2/', help='Generated results path.')
        self._parser.add_argument('--result_save', type=str, default='tree_pc_', help='Generated results name to save.')
        self._parser.add_argument('--visdom_port', type=int, default=8097, help='Visdom port number. (default:8097)')
        self._parser.add_argument('--visdom_color', type=int, default=4, help='Number of colors for visdom pointcloud visualization. (default:4)')

        # Network arguments
        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
        self._parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
        # NOTE: degree changed
        self._parser.add_argument('--degrees_opt', type=str, default='default', help='Upsample degrees for generator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
        #jz default is default=[3,  64,  128, 256, 512, 1024]
        # for D in r-GAN
        self._parser.add_argument('--D_FEAT', type=int, default=[3, 64,  128, 256, 256, 512], nargs='+', help='Features for discriminator.')
        # for D in source code
        # self._parser.add_argument('--D_FEAT', type=int, default=[3,  64,  128, 256, 512, 1024], nargs='+', help='Features for discriminator.')
        # Evaluation arguments
        self._parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')
        self._parser.add_argument('--ratio_base', type=int, default=500, help='# to select for each cat')
        
        self._parser.add_argument('--uniform_loss',type=str,default='None')
        self._parser.add_argument('--uniform_loss_scalar',type=float,default=20)
        self._parser.add_argument('--uniform_loss_radius',type=float,default=0.05)
        self._parser.add_argument('--uniform_loss_warmup_till',type=int,default=0) # if 0, no warmup, if 100, linear
        self._parser.add_argument('--uniform_loss_warmup_mode',type=int,default=1) # if 1, linear; if 0 step
        self._parser.add_argument('--radius_version', type=str,default='0')
        self._parser.add_argument('--uniform_loss_n_seeds', type=int,default=20)
        self._parser.add_argument('--uniform_loss_no_scale', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--uniform_loss_max', default=False, type=lambda x: (str(x).lower() == 'true'))    
        self._parser.add_argument('--uniform_loss_offset', type=int,default=0)
        self._parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--n_samples_train',type=int,default=0,help='# pcd to train')
        self._parser.add_argument('--eval_every_n_epoch',type=int,default=5)
        self._parser.add_argument('--patch_repulsion_loss',default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--n_sigma',type=float,default=1)
        self._parser.add_argument('--knn_loss',default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--knn_k',type=int,default=10)
        self._parser.add_argument('--knn_n_seeds',type=int,default=20)
        self._parser.add_argument('--knn_scalar',type=float,default=1)
        self._parser.add_argument('--krepul_loss',default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--krepul_k',type=int,default=10)
        self._parser.add_argument('--krepul_n_seeds',type=int,default=20)
        self._parser.add_argument('--krepul_scalar',type=float,default=1)
        self._parser.add_argument('--krepul_h',type=float,default=0.01)
        self._parser.add_argument('--expansion_penality',default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--expan_primitive_size',type=int,default=64)
        self._parser.add_argument('--expan_alpha',type=float,default=1.5)
        self._parser.add_argument('--expan_scalar',type=float,default=0.1)

        # conditonal GAN
        self._parser.add_argument('--n_classes',type=int, default=16)
        self._parser.add_argument('--cgan_version', type=int, default=0)
        self._parser.add_argument('--conditional', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--apex_flag', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--ddp_flag', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    def parser(self):
        return self._parser