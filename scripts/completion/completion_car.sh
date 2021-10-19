python trainer.py \
--dataset CRN \
--class_choice car \
--inversion_mode completion \
--mask_type k_mask \
--save_inversion_path ./saved_results/CRN_car_completion \
--ckpt_load pretrained_models/car.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/