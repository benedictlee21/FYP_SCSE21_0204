python eval_treegan.py \
--eval_treegan_mode generate_fpd_stats \
--class_choice cabinet \
--FPD_path ./evaluation/pre_statistics_CRN_cabinet.npz \
--save_sample_path ./saved_results/eval_treegan_samples \
--model_pathname ./pretrained_models/cabinet.pt \
--dataset_path ./input_shapes/shapenet_crn_dataset/
