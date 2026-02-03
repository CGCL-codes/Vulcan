#!/bin/bash
python /workspace/project/edgeseed/main.py \
  --model_name mask_rcnn_swin_tiny \
  --dataset_name coco \
  --task_type detection \
  --task_name T3-1 \
  --pruning_rate 0.2 \
  --penalty_param 1.0 \
  --learning_rate 5e-5 \
  --train_batch_size 8 \
  --eval_batch_size 32