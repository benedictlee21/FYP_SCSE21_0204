# 1. Generate multiclass pretrained model file (.pt) using 'pretrain_treegan.py'.

# 2. Generate multiclass FPD statistics file (.npz) using the multiclass model.
python eval_treegan.py \
--split train \
--eval_treegan_mode generate_fpd_stats \
--class_choice multiclass \
--save_sample_path ./saved_results/eval_treegan/ \
--model_pathname ./multiclass_pretrained_models/multiclass_chair_table_1000_epochs.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/ \
--class_range chair,table

# 3. Calculate FPD metric for multiclass pretrained model using statistics file.
python eval_treegan.py \
--eval_treegan_mode FPD \
--class_choice multiclass \
--conditional_gan True \
--FPD_path ./evaluation/pre_statistics_CRN_multiclass_chair_table.npz \
--save_sample_path ./saved_results/eval_treegan/ \
--model_pathname ./multiclass_pretrained_models/multiclass_chair_table_1000_epochs.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/ \
--class_range chair,table