python trainer.py \
--dataset CRN \
--class_choice couch \
--inversion_mode jittering \
--iterations 30 30 30 30 \
--visualize \
--save_inversion_path ./saved_results/CRN_couch_jittering \
--ckpt_load pretrained_models/couch.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/