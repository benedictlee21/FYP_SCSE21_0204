python trainer.py \
--dataset CRN \
--class_choice multiclass \
--inversion_mode multiclass \
--visualize \
--save_inversion_path ./saved_results/CRN_multiclass/ \
--ckpt_load ./multiclass_pretrained_models/multiclass_chair_table_1000_epochs.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/ \
--class_range chair,table
--batch_size 4