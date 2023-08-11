work_path=`pwd`
work_path=`basename $work_path`
PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p A10080G-share -n8 --gres=gpu:8  --ntasks-per-node=8 --cpus-per-task=14 \
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
  --start-point 5 \
  --end-point 6 \
  --with-timeline




