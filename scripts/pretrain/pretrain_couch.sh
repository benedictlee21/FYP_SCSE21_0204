python pretrain_treegan.py \
--split train \
--class_choice couch \
--FPD_path ./evaluation/pre_statistics_couch.npz \
--ckpt_path ./pretrain_checkpoints/couch \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/