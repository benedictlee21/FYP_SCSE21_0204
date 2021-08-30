srun -u --partition=Sensetime -n1 --gres=gpu:1 python verify_cd.py

### test benchmarks
python main.py --mode 1 --model_name topnet --load_model /mnt/lustre/zhangjunzhe/pcd_lib/pcd_benchmark/pretrained_models/topnet/best_cd_t_network.pth

python main.py --mode 1 --model_name cascade --load_model /mnt/lustre/zhangjunzhe/pcd_lib/pcd_benchmark/pretrained_models/cascade/best_cd_t_network.pth

python main.py --mode 1 --model_name pcn --load_model /mnt/lustre/zhangjunzhe/pcd_lib/pcd_benchmark/pretrained_models/pcn/best_cd_t_network.pth

python main.py --mode 1 --model_name msn --load_model /mnt/lustre/zhangjunzhe/pcd_lib/pcd_benchmark/pretrained_models/msn/best_cd_t_network.pth

### Aug full 
srun -u --partition=Sensetime -n1 --gres=gpu:8 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=knn6m_f \
        python train_single.py \
        --class_choice Chair --epochs 2000   --batch_size 512  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/knn6m_f/ \
        --n_samples_train 0 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --dataset cascade \
        --lr 1e-4 \
        --ckpt_load ./model/knn6m_f/tree_ckpt_30_Chair.pt

### Aug full all class
srun -u --partition=Sensetime -n1 --gres=gpu:8 --ntasks-per-node=1 \
        --job-name=knn6m_n \
        python train_single.py \
        --class_choice Chair --epochs 2000  \
        --batch_size 512  \
        --FPD_path ./evaluation/pre_statistics_all.npz \
        --ckpt_path ./model/knn6m_n/ \
        --n_samples_train 0 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --dataset cascade \
        --lr 1e-4 \
        --class_choice None 
        \
        --ckpt_load ./model/knn6m_n/tree_ckpt_10_None.pt

# ### retrain knn6m
srun -u --partition=Sensetime -n1 --gres=gpu:8 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=pc8b \
        python train_single.py \
        --class_choice Chair --epochs 20000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/knn6mb/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --ddp_flag False


# ### retrain knn6mc - train.py check it same or not
# srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
#         --job-name=knn6mc \
#         python train.py \
#         --class_choice Chair --epochs 2000   --batch_size 64  \
#         --FPD_path ./evaluation/pre_statistics_chair.npz \
#         --ckpt_path ./model/knn6mc/ \
#         --n_samples_train 1000 \
#         --eval_every_n_epoch 10 \
#         --knn_loss True \
#         --knn_n_seeds 100 \
#         --knn_k 30 \
#         --knn_scalar 0.2

### retrain knn6md - train_single.py ***** to test results good or not
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=knn6md \
        python train_single.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/knn6md/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --ckpt_load ./model/knn6md/tree_ckpt_540_Chair.pt

#### train on cascade h5 - airplane 1000
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=trial \
        python train_single.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/temp/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --dataset cascade

#### train on cascade h5 - chair 1000
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=knn6m_h5 \
        python train_single.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/knn6m_h5/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --dataset cascade

### kmm6m_64x4
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name= kmm6m_64x4\
        python train_single.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/kmm6m_64x4/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --dataset cascade 
        
        \
        --pretrain_D_lr 1e-4 \
        --pretrain_G_lr 3e-4




# speed h5
srun -u --partition=Sensetime -n1 --gres=gpu:8 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=trail \
        python train_single.py \
        --class_choice Chair --epochs 2000   --batch_size 512  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/trial/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --dataset cascade \
        --lr 2e-4
         \
        --lr 5e-4 \
        --ckpt_load ./model/knn6m_lb/tree_ckpt_290_Chair.pt


# apex
srun -u --partition=Sensetime -n1 --gres=gpu:8 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=knn6m_apex \
        python train_single.py \
        --class_choice Chair --epochs 2000   --batch_size 512  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/knn6m_apex/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --dataset cascade \
        --lr 5e-4 \
        --apex_flag True

## 对比
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=knn6m_h5 \
        python train_single.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/knn6m_h5/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --dataset cascade


# ddp
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-32 \
        --job-name=knn6m_ddp \
        python -m torch.distributed.launch --nproc_per_node=4 train_single.py \
        --class_choice Chair --epochs 2000   --batch_size 32  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/knn6m_ddp/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 20 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2 \
        --dataset cascade \
        --lr 5e-4 \
        --ddp_flag True


### eval and draw knn6md 350
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=eval \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --class_choice Chair \
        --load_model_path ./model/knn6mc \
        --save_sample_path ./saved_samples/temp \
        --epoch_load 440 \
        --batch_size 50 \
        --n_samples 1000 \
        --mmd_cov_jsd_ref_num 10 \
        --mmd_cov_jsd_sample_num 10 \
        --degrees_opt default \
        --test_modes F


### trial 
python -m torch.utils.bottleneck test_input_data.py
CUDA_VISIBLE_DEVICES=0,1 

srun -u --partition=Sensetime -n1 --gres=gpu:8 --ntasks-per-node=1 \
        python -m torch.distributed.launch --nproc_per_node=8 test_input_data.py


#!/bin/bash

partition=${1}
model_name=${2}
gpus=${3:-1}

root_dir=/mnt/lustre/$(whoami)/McDALF
config_dir=${root_dir}/mt_models/${model_name}
g=$((${gpus}<8?${gpus}:8))

export PYTHONPATH=${root_path}:$PYTHONPATH

srun -u --partition=${partition} --job-name=${model_name} \
    -n1 --gres=gpu:${gpus} --ntasks-per-node=1 \
    -x SG-IDC1-10-51-0-34 \
    python demo.py --name "bird_net" --num_train_epoch 500 --img_path "misc/demo_data/img1.jpg"



rm -r /mnt/lustre/zhangjunzhe/SinGAN/TrainedModels/birds  && 
srun -u --partition=Sensetime -n1 --gres=gpu:1  python gi_train.py 

srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python train.py --class_choice Airplane --batch_size 1000
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python train.py --class_choice Airplane




# new eval
# srun -u --partition=ips_share -n1 --gres=gpu:1 --ntasks-per-node=1 \
#         --job-name=eval \
#         python eval_GAN.py \
#         --dataset_path /mnt/lustre/cache/zhangjunzhe/datasets/shapenetcore_partanno_segmentation_benchmark_v0 \
#         --FPD_path ./evaluation/pre_statistics_chair.npz \
#         --class_choice Chair \
#         --load_model_path ./model/uni_loss_v3 \
#         --save_sample_path ./saved_samples/uni_loss_v3 \
#         --batch_size 100 \
#         --n_samples 500 \
#         --mmd_cov_jsd_ref_num 10 \
#         --mmd_cov_jsd_sample_num 10 \
#         --epoch_load 1660 \
#         --test_modes FJM
        
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=eval \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --class_choice Airplane \
        --load_model_path ./model/cus_los0a \
        --save_sample_path ./saved_samples/cus_los0a \
        --epoch_load 1830 \
        --batch_size 50 \
        --n_samples 50 \
        --mmd_cov_jsd_ref_num 10 \
        --mmd_cov_jsd_sample_num 10 \
        --degrees_opt default \
        --test_modes S








srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=eval \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --class_choice Chair \
        --load_model_path ./model/uni_loss_v35 \
        --save_sample_path ./saved_samples/branch_v3 \
        --batch_size 100 \
        --n_samples 100 \
        --mmd_cov_jsd_ref_num 10 \
        --mmd_cov_jsd_sample_num 10 \
        --epoch_load 1310 \
        --degrees_opt opt_3 \
        --test_modes S


# eval gan 4.5  gen_gan_4.5_2nd 
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=eval_fpd \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_4x500.npz \
        --class_choice Chair \
        --save_num_generated 2000 \
        --gen_path ./gen_temp \
        --model_path ./model/gen_gan_4.1_2nd \
        --n_classes 4 

        --conditional True \
        --conditional_ratio True \
        
        # --version 0



# June 8, train treegan, with uniform loss
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_test \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 10  \
        --ckpt_path ./model/checkpoints2/ \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_load ./model/checkpoints17/tree_ckpt_1620_Chair.pt \
        --uniform_loss uniform_v1



srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=cus_los28 \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_plane.npz \
        --ckpt_path ./model/cus_los28/ \
        --uniform_loss two_custom_loss \
        --uniform_loss_scalar 0.5 \
        --uniform_loss_radius  0.05 \
        --uniform_loss_n_seeds 200 \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --radius_version cus_los28
        
        \
        --ckpt_load ./model/cus_los26/tree_ckpt_120_Chair.pt

# expan
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=expan6 \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/expan6/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --expansion_penality True \
        --expan_alpha 1.5 \
        --expan_scalar 0.1 \

        # krepul
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 10 \
        --knn_scalar 0.2
        # with cus los
        --uniform_loss custom_loss \
        --uniform_loss_scalar 0.2 \
        --uniform_loss_radius  0.05 \
        --uniform_loss_n_seeds 200 


        --ckpt_load ./model/expan3/tree_ckpt_1280_Chair.pt

# krepul loss
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=krepul6 \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/krepul6/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --krepul_loss True \
        --krepul_n_seeds 100 \
        --krepul_k 10 \
        --krepul_h 0.03 \
        --krepul_scalar 0.2


# kNN loss
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=knn8 \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/knn8/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 10 \
        --knn_scalar 0.04

        --ckpt_load ./model/cus_los7/tree_ckpt_1970_Chair.pt
        --ckpt_load ./model/uni_loss_v0/tree_ckpt_1650_Chair.pt



# patch_repulsion_loss
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=repul12 \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/repul12/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --patch_repulsion_loss True  \
        --n_sigma 0.4 \
        --uniform_loss custom_loss \
        --uniform_loss_scalar 2000 \
        --uniform_loss_radius  0.02 \
        --uniform_loss_n_seeds 200 
        
        
        \
        --uniform_loss custom_loss \
        --uniform_loss_scalar 0.1 \
        --uniform_loss_radius  0.05 \
        --uniform_loss_n_seeds 200 


        --ckpt_load ./model/checkpoints17/tree_ckpt_1620_Chair.pt  
        
       
        

# June 13 train diff degrees
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=branch_v6 \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/branch_v6/ \
        --degrees_opt opt_1 \
        --loop_non_linear True \
        --uniform_loss uniform_single_radius \
        --uniform_loss_scalar 1  \
        --uniform_loss_radius  0.01 \
        --uniform_loss_n_seeds 200 \
        --uniform_loss_offset 5 \
        --uniform_loss_no_scale True \
        --uniform_loss_max True

#june 13 train different loop term
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=loop \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/loop/ \
        --loop_non_linear True

# june 15, train small data
#june 13 train different loop term
srun -u --partition=Sensetime -n1 --gres=gpu:8 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=sml_v4 \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 256  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/sml_v4/ \
        --n_samples_train 0
=============================================================


# June 1, train cgan, 4.5
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=ctgv0_4.5 \
        python train_cgan.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path ./model/gen_cgan_v0_4.5/ \
        --FPD_path ./evaluation/pre_statistics_all.npz \
        --n_classes 4 
        
        \
        --ckpt_load ./model/gen_cgan_v0_4.5/tree_ckpt_35_None.pt

# June 1, train cgan, 4.5 v1
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=ctg_v1 \
        python train_cgan.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path ./model/gen_cgan_v1/ \
        --FPD_path ./evaluation/pre_statistics_all.npz \
        --n_classes 4 \
        --version 1


# June 1, train cgan, 4.5 v2
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=ctg_v2 \
        python train_cgan.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path ./model/gen_cgan_v2/ \
        --FPD_path ./evaluation/pre_statistics_all.npz \
        --n_classes 4 \
        --version 2

# June 1, train cgan, 4.5 v3
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=ctg_v3 \
        python train_cgan.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path ./model/gen_cgan_v3/ \
        --FPD_path ./evaluation/pre_statistics_all.npz \
        --n_classes 4 \
        --version 3



# June 1, train cgan, 4.5 v4
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=ctg_v4 \
        python train_cgan.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path ./model/gen_cgan_v4/ \
        --FPD_path ./evaluation/pre_statistics_all.npz \
        --n_classes 4 \
        --version 4


# trail using cgan 4.5 model to test against pre 4 npz
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=ctg_tv1 \
        python train_cgan.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path ./model/checkpoints2/ \
        --FPD_path ./evaluation/pre_statistics_all.npz \
        --n_classes 4 \
        --version 1 \
        --FPD_path ./evaluation/pre_statistics_4x500.npz \
        --ckpt_load ./model/gen_cgan_v1/tree_ckpt_400_None.pt

# trial using gan 4.5 model to test against pre 4 npz
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_test \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 64  \
        --ratio_base 500 \
        --dataset ShapeNet_v0  \
        --ckpt_path ./model/checkpoints2/ \
        --FPD_path ./evaluation/pre_statistics_4x500.npz \
        --ckpt_load ./gen_gan_4.5tree_ckpt_1155_None.pt

# June 1, train train gan using rGAN data
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=rgan_chair.5c \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --ratio_base 500 \
        --dataset ShapeNet_v0_rGAN_Chair  \
        --ckpt_path ./model/gen_rgan_chair/ \
        --FPD_path ./evaluation/pre_statistics_chair.npz

# May 31, train gan, 4 class 500, 
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_4.5c \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 64  \
        --ratio_base 500 \
        --dataset ShapeNet_v0  \
        --ckpt_path ./model/gen_gan_4.5/ \
        --FPD_path ./evaluation/pre_statistics_all.npz

# May 31, train gan, 4 class 1000, 
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_4.1c \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 64  \
        --ratio_base 1000 \
        --dataset ShapeNet_v0  \
        --ckpt_path ./model/gen_gan_4.1/ \
        --FPD_path ./evaluation/pre_statistics_all.npz



# tg_chair script, on 430pm May 14 (batch 64)

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg18chair \
        python train.py \
        --class_choice Chair --epochs 2000 \
        --FPD_path ./evaluation/pre_statistics_chair.npz

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg18plane \
        python train.py \
        --class_choice Airplane --epochs 2000 \
        --FPD_path ./evaluation/pre_statistics_plane.npz

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg18all \
        python train.py \
        --class_choice None --epochs 2000 \
        --FPD_path ./evaluation/pre_statistics_all.npz



srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_chair \
        python train.py \
        --class_choice Chair --epochs 2000 --batch_size 20

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_chair \
        python train.py \
        --class_choice Chair --epochs 2000  \
        --ckpt_load tree_ckpt_420_Chair.pt


# tg_all 
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_all \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 64                

# tg_sofa ( actually it is table)
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_table \
        python train.py \
        --class_choice Table --epochs 2000    --batch_size 20  


srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_cap \
        python train.py \
        --class_choice Cap --epochs 2000    --batch_size 20  
       

srun -u --partition=Sensetime -n1 --job-name=allcls2  --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python train.py --class_choice None --epochs 2000


# preprocess for all class, but not uniform
srun -u --partition=Sensetime -n1 --job-name=allcls2  --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python dataset_preparation.py --class_choice None