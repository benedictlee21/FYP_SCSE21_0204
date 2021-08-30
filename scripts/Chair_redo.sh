#knn6m -- save module
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=knn6m \
        python train.py \
        --class_choice Chair --epochs 2000   --batch_size 64  \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --ckpt_path ./model/knn6m/ \
        --n_samples_train 1000 \
        --eval_every_n_epoch 10 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2