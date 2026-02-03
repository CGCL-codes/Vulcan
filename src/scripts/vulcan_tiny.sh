#!/bin/bash
python /workspace/project/edgeseed/main.py \
  --model_name deit_tiny_patch16_224 \
  --dataset_name cifar10 \
  --task_name T1-2 \
  --pruning_rate 0.9 \
  --penalty_param 1.0