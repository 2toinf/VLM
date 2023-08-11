PARTITION=$1

PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p ${PARTITION} -n8 --ntasks-per-node=8 --gres=gpu:8  --cpus-per-task=5 \
python -u /mnt/lustre/zhengjinliang/vision-language-model/VQ-VAE/train.py \
    --output_dir log \
    --log_dir log \
    --process_type default \
    --train_interpolation bicubic \
    --min_crop_scale 0.08 \
    --model vqkd_encoder_base_decoder_3x768x12_pixel \
    --teacher_input_size 224 \
    --codebook_n_emd 16384  \
    --codebook_emd_dim 32 \
    --quantize_kmeans_init \
    --rec_loss_type mse \
    --batch_size 64 \
    --opt adamw \
    --opt_betas 0.9 0.99 \
    --weight_decay 1e-4  \
    --warmup_epochs 10 \
    --epochs 100 \
    --save_ckpt_freq 20 