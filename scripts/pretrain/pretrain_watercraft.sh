python pretrain_treegan.py \
--split train \
--class_choice watercraft \
--FPD_path ./evaluation/pre_statistics_watercraft.npz \
--ckpt_path ./pretrain_checkpoints/watercraft \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/