work_path=`pwd`
work_path=`basename $work_path`
PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p DBX32 -n8 --gres=gpu:8  --ntasks-per-node=8 --cpus-per-task=5 \
python -u /mnt/lustre/zhengjinliang/vision-language-model/frame_predictor/main_frame_predict.py \
  --model translator_kl_8_256 \
  --input-size 256 \
  --vqvae vq-8-n256 \
  --batch-size 32 \
  --output_dir ckpt \
  --epochs 300 \
  --num_workers 8 \
  --port 29522 \
  --resume checkpoint.pth \
  --loss-type kl \
  --start-point 7 \
  --end-point 8 \
  --with-timeline




