python pretrain_treegan.py \
--split train \
--class_choice table \
--FPD_path ./evaluation/pre_statistics_table.npz \
--ckpt_path ./pretrain_checkpoints/table \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/