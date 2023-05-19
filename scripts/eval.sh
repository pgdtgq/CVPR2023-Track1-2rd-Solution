export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s
# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s

config=./configs/eval.py

python3 -m paddle.distributed.launch --log_dir=./logs/eval --gpus="0,1,2,3,4,5,6,7" ./tools/ufo_eval.py --config-file ${config} --eval-only --aug_eval_seg --flip_horizontal_seg


