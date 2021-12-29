python pretrain_treegan.py \
--split train \
--class_choice multiclass \
--FPD_path ./evaluation/pre_statistics_CRN_multiclass.npz \
--ckpt_path ./multiclass_pretrained_models/ \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/ \
--class_range chair,table \
--epochs 1000 \
--batch_size 4 \
--samples_per_class 1000 \
--eval_every_n_epoch 0 \
--save_every_n_epoch 50