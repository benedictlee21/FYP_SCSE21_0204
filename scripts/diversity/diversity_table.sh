python trainer.py \
--dataset CRN \
--class_choice table \
--inversion_mode diversity \
--visualize \
--save_inversion_path ./saved_results/CRN_table_diversity \
--ckpt_load ./pretrained_models/table.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/