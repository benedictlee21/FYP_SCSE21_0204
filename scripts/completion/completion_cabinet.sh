python trainer.py \
--dataset CRN \
--class_choice cabinet \
--inversion_mode completion \
--mask_type k_mask \
--save_inversion_path ./saved_results/CRN_cabinet_completion \
--ckpt_load pretrained_models/cabinet.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/