


srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python completion_cascade.py \
        --batch_size 10 \
        --n_samples_train 1000 \
        --ftr_type Discriminator \
        --update_G True \
        --save_inversion_path ./saved_samples/invert_topnet_scan2 \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt \
        --dgp_mode topnet \
        --n_bins 8 \
        --max_bins 16 \
        --target_resol_bins 16 \
        --init_by_ftr_loss False \
        --increase_n_bins True \
        --downsample True \
        --cd_option one_sided \
        --select_num 500 \
        --class_choice  chair \
        --start_point 10 \
        --run_len 5











srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 \
        --job-name=trial \
        python completion_cascade.py \
        --batch_size 10 \
        --n_samples_train 1000 \
        --ftr_type Discriminator \
        --update_G True \
        --save_inversion_path ./saved_samples/invert_topnet_scan2 \
        --ckpt_load ./model/knn6m/tree_ckpt_1870_Chair.pt \
        --dgp_mode ball_hole \
        --hole_radius 0.30 \
        --hole_n 3 \
        --given_partial False \
        --class_choice  chair \
        --start_point 10 \
        --run_len 5



# opt.start_point,run_len 10 5
# print init x_map and target cd 0.002000167965888977
# stage 0 -- 0.0005041788681410253
# stage 1 -- 0.0003991233534179628
# stage 2 -- 0.00033559766598045826
# stage 3 -- 0.00032263004686683416
# save pcs done
# 10 done,  43 s
# print init x_map and target cd 0.00015567161608487368
# stage 0 -- 0.0010891025885939598
# stage 1 -- 0.0017479434609413147
# stage 2 -- 0.0018081909511238337
# stage 3 -- 0.0015660423086956143
# save pcs done
# 11 done,  38 s
# print init x_map and target cd 0.0005152317462489009
# stage 0 -- 0.0004654050571843982
# stage 1 -- 0.00040312076453119516
# stage 2 -- 0.000363112980267033
# stage 3 -- 0.00032949697924777865
# save pcs done
# 12 done,  41 s
# print init x_map and target cd 0.0009738322696648538
# stage 0 -- 0.0012339736567810178
# stage 1 -- 0.0007597727817483246
# stage 2 -- 0.0005791262374259531
# zstage 3 -- 0.0005544298328459263
# save pcs done
# 13 done,  38 s
# print init x_map and target cd 0.0003939977614209056
# stage 0 -- 0.004197497386485338
# stage 1 -- 0.0034135060850530863
# stage 2 -- 0.0034467189107090235
# stage 3 -- 0.0035546047147363424