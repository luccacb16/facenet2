cmd="python3 train.py --batch_size 32 --accumulation 512 --epochs 20 --margin 0.4 --num_workers 1 --data_path ./data/ --checkpoint_path '../drive/MyDrive/FaceNet_checkpoints/' --colab True"
echo $cmd
$cmd