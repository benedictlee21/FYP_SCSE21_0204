python eval_treegan.py \
--eval_treegan mode generate_fpd_stats (or <FPD>) \
--class_choice chair \
--FPD_path ./evaluation/pre_statistics_CRN_multiclass.npz \
--save_sample_path ./saved_results/eval_treegan/ \
--model_pathname ./pretrained_checkpoints/multiclass.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/
--class_range chair,table