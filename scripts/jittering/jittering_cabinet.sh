python trainer.py \
--dataset CRN \
--class_choice cabinet \
--inversion_mode jittering \
--iterations 30 30 30 30 \
--visualize \
--save_inversion_path ./saved_results/CRN_cabinet_jittering \
--ckpt_load pretrained_models/cabinet.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/