python trainer.py \
--dataset CRN \
--class_choice chair \
--inversion_mode multiclass \
--visualize \
--save_inversion_path ./saved_results/CRN_multiclass \
--ckpt_load ./pretrained_models/chair.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset \
--class_range chair,table,couch,cabinet,lamp,car,plane,watercraft