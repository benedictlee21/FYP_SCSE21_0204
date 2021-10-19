python trainer.py \
--dataset CRN \
--class_choice watercraft \
--inversion_mode diversity \
--visualize \
--save_inversion_path ./saved_results/CRN_watercraft_diversity \
--ckpt_load ./pretrained_models/watercraft.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/