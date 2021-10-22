python pretrain_treegan.py \
--split train \
--class_choice chair \
--FPD_path ./evaluation/pre_statistics_chair.npz \
--ckpt_path ./pretrain_checkpoints/multi/ \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/ \
--class_range chair,table