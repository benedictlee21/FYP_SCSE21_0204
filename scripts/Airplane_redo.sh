
#knn6a
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=knn6a \
        python train.py \
        --class_choice Airplane --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_plane.npz \
        --ckpt_path ./model/knn6a/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2

#expan9a
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=expan8a \
        python train.py \
        --class_choice ChAirplaneair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_plane.npz \
        --ckpt_path ./model/expan8a/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --expansion_penality True \
        --expan_alpha 1.5 \
        --expan_scalar 100 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2


#expan8a
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=expan8a \
        python train.py \
        --class_choice ChAirplaneair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_plane.npz \
        --ckpt_path ./model/expan8a/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --expansion_penality True \
        --expan_alpha 1.5 \
        --expan_scalar 0.1 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2


#expan6a
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=expan6a \
        python train.py \
        --class_choice ChAirplaneair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_plane.npz \
        --ckpt_path ./model/expan6a/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --expansion_penality True \
        --expan_alpha 1.5 \
        --expan_scalar 0.1 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 10 \
        --knn_scalar 0.2

#cus_los28a
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=cus_los28a \
        python train.py \
        --class_choice Airplane --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_plane.npz \
        --ckpt_path ./model/cus_los28a/ \
        --uniform_loss two_custom_loss \
        --uniform_loss_scalar 0.5 \
        --uniform_loss_radius  0.05 \
        --uniform_loss_n_seeds 200 \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --radius_version cus_los28

#cus_los0a
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=cus_los0a \
        python train.py \
        --class_choice Airplane --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_plane.npz \
        --ckpt_path ./model/cus_los0a/ \
        --uniform_loss custom_loss \
        --uniform_loss_scalar 0.2 \
        --uniform_loss_radius  0.05 \
        --uniform_loss_n_seeds 200 \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 