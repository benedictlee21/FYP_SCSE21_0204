python trainer.py \
--dataset CRN \
--class_choice chair \
--inversion_mode diversity \
--visualize \
--save_inversion_path ./saved_results/CRN_chair_diversity \
--ckpt_load ./pretrained_models/chair.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/