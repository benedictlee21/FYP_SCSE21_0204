python trainer.py \
--dataset CRN \
--class_choice chair \
--inversion_mode jittering \
--iterations 30 30 30 30 \
--visualize \
--save_inversion_path ./saved_results/CRN_chair_jittering \
--ckpt_load pretrained_models/chair.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/