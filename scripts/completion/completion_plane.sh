python trainer.py \
--dataset CRN \
--class_choice plane \
--inversion_mode completion \
--mask_type k_mask \
--save_inversion_path ./saved_results/CRN_plane_completion \
--ckpt_load pretrained_models/plane.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/