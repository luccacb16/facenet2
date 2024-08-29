cmd="python3 train.py --batch_size 32 --accumulation 256 --epochs 20 --margin 0.7 --num_workers 1 --data_path ./data/ --checkpoint_path '../drive/MyDrive/FaceNet_checkpoints/' --colab True"
echo $cmd
$cmd