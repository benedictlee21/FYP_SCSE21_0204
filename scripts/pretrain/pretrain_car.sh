python pretrain_treegan.py \
--split train \
--class_choice car \
--FPD_path ./evaluation/pre_statistics_car.npz \
--ckpt_path ./pretrain_checkpoints/car \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/