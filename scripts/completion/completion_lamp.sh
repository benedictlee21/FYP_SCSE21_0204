python trainer.py \
--dataset CRN \
--class_choice lamp \
--inversion_mode completion \
--mask_type k_mask \
--save_inversion_path ./saved_results/CRN_lamp_completion \
--ckpt_load pretrained_models/lamp.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/