
#cg0
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=cg0 \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path ./model/cg0/ \
        --FPD_path ./evaluation/pre_statistics_4x500.npz \
        --n_classes 4 \
        --conditional True \
        --dataset ShapeNet_v0 \
        --cgan_version 0


#cg0 - expan6a --> cg1
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=cg1 \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path ./model/cg1/ \
        --FPD_path ./evaluation/pre_statistics_4x500.npz \
        --n_classes 4 \
        --conditional True \
        --dataset ShapeNet_v0 \
        --cgan_version 0 \
        --expansion_penality True \
        --expan_alpha 1.5 \
        --expan_scalar 0.1 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 10 \
        --knn_scalar 0.2


#cg0 - expan8a --> cg2
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=cg2 \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path ./model/cg2/ \
        --FPD_path ./evaluation/pre_statistics_4x500.npz \
        --n_classes 4 \
        --conditional True \
        --dataset ShapeNet_v0 \
        --cgan_version 0 \
        --expansion_penality True \
        --expan_alpha 1.5 \
        --expan_scalar 0.1 \
        --knn_loss True \
        --knn_n_seeds 100 \
        --knn_k 30 \
        --knn_scalar 0.2


        


        