#!/bin/bash
python /workspace/project/edgeseed/main.py \
  --model_name deit_base_patch16_224 \
  --dataset_name imagenet \
  --task_name T5-50 \
  --pruning_rate 0.6 \
  --penalty_param 1.0