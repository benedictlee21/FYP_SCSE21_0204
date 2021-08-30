import argparse
from test import test
from train import pcn_train, topnet_train, cascade_train, msn_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Completion')

    # mode
    parser.add_argument('--mode', type=int, default=0, help='0 for train, 1 for test')
    parser.add_argument('--model_dir', type=str, default='') # for test only

    # common args
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--model_name', type=str, default='cascade',  help='pcn/topnet/msn/cascade')
    parser.add_argument('--load_model', type=str, default='',  help='load model to resume training / start testing')
    parser.add_argument('--resume_epoch', type=int, default=0, help='which epoch to resume from')
    parser.add_argument('--num_points', type=int, default=2048,  help='number of ground truth points')
    parser.add_argument('--log_env', type=str, default="cascade_2048", help='subfolder name inside log/<model>_<loss>_train/')
    parser.add_argument('--loss', type=str, default='EMD', help='train loss type; CD or EMD')
    parser.add_argument('--manual_seed', type=str, default='', help='manual seed')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')  # cascade, msn, pcn:0.0001, topnet:0.5e-2
    parser.add_argument('--lr_clip', type=float, default=1e-6, help='learning rate')

    # msn:
    parser.add_argument('--n_primitives', type=int, default=16,  help='number of surface elements') # 32 for 16384

    # cascade:
    parser.add_argument('--step_ratio', type=int, default=2)  # 2 if gt num point = 2048, 4:4096, 8:8192, 16:16384
    parser.add_argument('--use_mean_feature', type=int, default=0, help='0 if not using, 1 if using')

    args = parser.parse_args()

    models_dict = {'pcn': pcn_train.train,
                   'topnet': topnet_train.train,
                   'msn': msn_train.train,
                   'cascade': cascade_train.train}

    assert args.model_name in list(models_dict.keys())
    assert args.loss == 'EMD' or args.loss == 'CD'

    if args.mode == 0:
        models_dict[args.model_name](args)
    else:
        test(args)










