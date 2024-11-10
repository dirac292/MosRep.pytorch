set -x

pip install -r requirements.txt

# IMAGENET_TRAIN="IMAGENET_DIR/train/shards-{00000..01281}.tar"
# TRAIN="/mnt/pub1/ssl-pretraining/data/hyper-k-mosrep/shards-{00000..00099}.tar"
# IMAGENET_VAL="IMAGENET_DIR/val/shards-{00000..00049}.tar"
       # --moco-k 65536 \
# Batch Size = 8 GPUs * 64 = 256
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
       pretrain_ddp.py \
       -a "resnet50" \
       -m "mosrep" \
       -b 128 \
       -j 8 \
       --lr 0.03 \
       --epochs 10 \
       --multi-crop \
       --global-scale 0.2 1.0 \
       --global-size 224 \
       --local-scale 0.1 0.6 \
       --local-size 112 \
       --shift-enable 1.0 \
       --shift-pix 48 \
       --shift-beta 0.5 \
       --moco-k 57344 \
       --seed 42 \
       --dist-url "tcp://localhost:1234" \
       --exp-folder "EXP_FOLDER" \
       --exp-name "mosrep_in1k_lr0.03_bs256_100ep"
