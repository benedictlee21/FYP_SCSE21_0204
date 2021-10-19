python trainer.py \
--dataset CRN \
--class_choice lamp \
--inversion_mode diversity \
--visualize \
--save_inversion_path ./saved_results/CRN_lamp_diversity \
--ckpt_load ./pretrained_models/lamp.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/