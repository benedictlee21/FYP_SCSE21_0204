python trainer.py \
--dataset CRN \
--class_choice car \
--inversion_mode diversity \
--visualize \
--save_inversion_path ./saved_results/CRN_car_diversity \
--ckpt_load ./pretrained_models/car.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/