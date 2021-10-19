python pretrain_treegan.py \
--split train \
--class_choice plane \
--FPD_path ./evaluation/pre_statistics_plane.npz \
--ckpt_path ./pretrain_checkpoints/plane \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/