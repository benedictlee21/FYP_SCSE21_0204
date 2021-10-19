python pretrain_treegan.py \
--split train \
--class_choice lamp \
--FPD_path ./evaluation/pre_statistics_lamp.npz \
--ckpt_path ./pretrain_checkpoints/lamp \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/