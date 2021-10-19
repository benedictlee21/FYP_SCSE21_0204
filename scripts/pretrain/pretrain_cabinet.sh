python pretrain_treegan.py \
--split train \
--class_choice cabinet \
--FPD_path ./evaluation/pre_statistics_cabinet.npz \
--ckpt_path ./pretrain_checkpoints/cabinet \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/