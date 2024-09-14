cmd="python3 train.py --batch_size 32 --accumulation 512 --epochs 20 --margin 0.4 --num_workers 1 --data_path ./data/ --checkpoint_path ./checkpoints/"
echo $cmd
$cmd

cmd="python3 train.py \
  --model FaceResNet50 \
  --batch_size 128 \
  --accumulation 1024 \
  --change_mining_step 5 \
  --warmup_epochs 5 \
  --epochs 40 \
  --margin 0.2 \
  --emb_size 512 \
  --min_lr 1e-6 \
  --max_lr 1e-3 \
  --num_workers 4 \
  --data_path '../../input/casia119k/' \
  --checkpoint_path checkpoints/ \
  --wandb \
  --restore pretrained.pt"

echo $cmd
$cmd