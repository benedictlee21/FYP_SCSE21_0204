python trainer.py \
--dataset CRN \
--class_choice watercraft \
--inversion_mode jittering \
--iterations 30 30 30 30 \
--visualize \
--save_inversion_path ./saved_results/CRN_watercraft_jittering \
--ckpt_load pretrained_models/watercraft.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/