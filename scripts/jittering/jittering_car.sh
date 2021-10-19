python trainer.py \
--dataset CRN \
--class_choice car \
--inversion_mode jittering \
--iterations 30 30 30 30 \
--visualize \
--save_inversion_path ./saved_results/CRN_car_jittering \
--ckpt_load pretrained_models/car.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/