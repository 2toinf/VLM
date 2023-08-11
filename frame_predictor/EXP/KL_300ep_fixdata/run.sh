work_path=`pwd`
work_path=`basename $work_path`
PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p DBX32 -n8 --gres=gpu:8  --ntasks-per-node=8 --cpus-per-task=5 \
python -u /mnt/lustre/zhengjinliang/vision-language-model/frame_predictor/main_frame_predict.py \
  --model translator_kl_large \
  --input-size 128 \
  --batch-size 64 \
  --vqvae vq-8-n256 \
  --output_dir ckpt \
  --epochs 300 \
  --num_workers 8 \
  --port 29522 \
  --resume checkpoint.pth \
  --loss-type kl \
  --start-point 4 \
  --end-point 5 \
  --with-timeline \
  --pretrained /mnt/lustre/zhengjinliang/vision-language-model/frame_predictor/EXP/KL_100ep_larger/checkpoint.pth




