python trainer.py \
--dataset CRN \
--class_choice plane \
--inversion_mode jittering \
--iterations 30 30 30 30 \
--visualize \
--save_inversion_path ./saved_results/CRN_plane_jittering \
--ckpt_load pretrained_models/plane.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/