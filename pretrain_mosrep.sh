set -x

pip install -r requirements.txt


# Batch Size = 2 GPUs * 64 = 128
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
       pretrain_ddp.py \
       -a "resnet50" \
       -m "mosrep" \
       -b 64 \
       -j 16 \
       --lr 0.015 \
       --epochs 10 \
       --multi-crop \
       --global-scale 0.2 1.0 \
       --global-size 224 \
       --local-scale 0.1 0.6 \
       --local-size 112 \
       --shift-enable 1.0 \
       --shift-pix 48 \
       --shift-beta 0.5 \
       --moco-k 65536 \
       --seed 42 \
       --dist-url "tcp://localhost:1234" \
       --exp-folder "EXP_FOLDER" \
       --exp-name "mosrep_hyperK_lr0.015_bs128_10ep"
