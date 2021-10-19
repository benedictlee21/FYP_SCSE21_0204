python trainer.py \
--dataset CRN \
--class_choice cabinet \
--inversion_mode diversity \
--visualize \
--save_inversion_path ./saved_results/CRN_cabinet_diversity \
--ckpt_load ./pretrained_models/cabinet.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/