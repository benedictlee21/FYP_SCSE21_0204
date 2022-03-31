python pretrain_treegan.py \
--split train \
--class_choice multiclass \
--FPD_path ./evaluation/pre_statistics_CRN_multiclass_chair_table.npz \
--ckpt_path ./multiclass_pretrained_models_cgan/ \
--knn_loss True \
--dataset_path ./input_shapes/shapenet_crn_dataset/ \
--class_range chair,table \
--conditional_gan True \
--epochs 1000 \
--batch_size 4 \
--samples_per_class 1000 \
--eval_every_n_epoch 0 \
--save_every_n_epoch 50

# For CUDA debugging, add statement:
# CUDA_LANUCH_BLOCKING="1"
# Before the command 'python'.