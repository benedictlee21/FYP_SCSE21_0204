
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=draw \
        python plots/completion_pcd_plot.py





/mnt/cache/zhangjunzhe/datasets/shapenetcore_partanno_segmentation_benchmark_v0


### in 198.6
srun -u --partition=vi_irdc_v100_32g -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python gi_train.py \
        --batch_size 10 \
        --n_samples_train 1000 \
        --ftr_type Discriminator \
        --update_G True \
        --save_inversion_path ./saved_samples/invert_topnet_scan2 \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt \
        --dgp_mode topnet \
        --dataset_path /mnt/cache/zhangjunzhe/datasets/shapenetcore_partanno_segmentation_benchmark_v0

######
# topnet
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python gi_train.py \
        --batch_size 10 \
        --n_samples_train 1000 \
        --ftr_type Discriminator \
        --update_G True \
        --save_inversion_path ./saved_samples/invert_topnet_scan2 \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt \
        --dgp_mode topnet \
        --n_bins 8 \
        --max_bins 16 \
        --target_resol_bins 16 \
        --init_by_ftr_loss False \
        --increase_n_bins True \
        --downsample True \
        --cd_option one_sided \
        --select_num 1000 

# ball hole

srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python gi_train.py \
        --batch_size 10 \
        --n_samples_train 1000 \
        --ftr_type Discriminator \
        --update_G True \
        --save_inversion_path ./saved_samples/invert_topnet_scan2 \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt \
        --dgp_mode ball_hole \
        --hole_radius 0.30 \
        --hole_n 3 \
        --given_partial False



# knn_hole
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python gi_train.py \
        --batch_size 10 \
        --n_samples_train 1000 \
        --ftr_type Discriminator \
        --update_G True \
        --save_inversion_path ./saved_samples/invert_knn_hole_5x200 \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt \
        --dgp_mode knn_hole \
        --hole_k 200 \
        --hole_n 5


# rec
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python gi_train.py \
        --batch_size 10 \
        --n_samples_train 1000 \
        --ftr_type Discriminator \
        --update_G True \
        --save_inversion_path ./saved_samples/invert_recon_no_emd \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt \
        --dgp_mode reconstruction \
        --hole_k 200 \
        --hole_n 5





srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python gi_train.py \
        --batch_size 10 \
        --n_samples_train 1000 \
        --ftr_type Discriminator \
        --update_G True \
        --save_inversion_path ./saved_samples/invert_knn_hole_5x200 \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt \
        --dgp_mode knn_hole \
        --hole_radius 0.25 \
        --hole_k 200 \
        --hole_n 5




#
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python gi_train.py \
        --batch_size 10 \
        --n_samples_train 1000 \
        --ftr_type Discriminator \
        --update_G True \
        --save_inversion_path ./saved_samples/invert_knn_hole_k500 \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt \
        --dgp_mode ball_hole \
        --hole_radius 0.25 \
        --hole_k  500


srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python gi_train.py \
        --batch_size 10 \
        --n_samples_train 10 \
        --ftr_type perceptual \
        --update_G True \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt


srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=eval \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --class_choice Chair \
        --load_model_path ./model/krepul5 \
        --save_sample_path ./saved_samples/krepul5 \
        --epoch_load 1950 \
        --batch_size 100 \
        --n_samples 5000 \
        --mmd_cov_jsd_ref_num 10 \
        --mmd_cov_jsd_sample_num 10 \
        --degrees_opt default \
        --test_modes SF


--ckpt_load ./model/temp/test.pt
srun -u --partition=Sensetime -n1 --gres=gpu:1  python test_input_data.py 

srun -u --partition=Sensetime -n1 --gres=gpu:2 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=temp \
        python train.py \
        --class_choice Airplane --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_plane.npz \
        --ckpt_path ./model/temp/ \
        --uniform_loss custom_loss \
        --uniform_loss_scalar 0.2 \
        --uniform_loss_radius  0.05 \
        --uniform_loss_n_seeds 200 \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 