export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s
# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s

config=./configs/train.py

python3 -m paddle.distributed.launch --log_dir=./logs/train --gpus="0,1,2,3,4,5,6,7" ./tools/ufo_train.py --config-file ${config} 


