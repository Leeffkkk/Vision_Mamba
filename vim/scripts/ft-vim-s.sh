#!/bin/bash
# conda activate nnrt_vim
cd /scratch/yufan/NNRTNormal/Vision_Mamba/vim;

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/leeff/.conda/envs/nnrt_vim_v3/bin/python\
 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py\
 --model vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
 --batch-size 16 \
 --lr 5e-6 \
 --min-lr 1e-5 \
 --warmup-lr 1e-5 \
 --drop-path 0.0 \
 --weight-decay 1e-8 \
 --num_workers 25 \
 --data-path /scratch/yufan/MPII_data/ \
 --data-set 'MPII' \
 --output_dir ./output/vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
 --epochs 100 \
 --finetune /scratch/yufan/NNRTNormal/Vision_Mamba/vim/output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2/best_checkpoint_8530.pth \
 --input-size 448 \
 --no_amp