#!/bin/bash
# conda activate seaicevim
cd /scratch/yufan/NNRTNormal/Vision_Mamba/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py\
 --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2\
 --batch-size 64 \
 --drop-path 0.05 \
 --weight-decay 0.05 \
 --lr 1e-3 \
 --num_workers 25 \
 --data-path /scratch/yufan/MPII_data/ \
 --data-set 'MPII' \
 --output_dir ./output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
 --epochs 1000 \
 --input-size 448 \
 --no_amp
