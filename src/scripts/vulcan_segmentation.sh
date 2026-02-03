#!/bin/bash
python /workspace/project/edgeseed/main.py \
  --model_name mask_rcnn_swin_tiny \
  --dataset_name coco \
  --task_type segmentation \
  --task_name T1-1 \
  --pruning_rate 0.8 \
  --penalty_param 1.0 \
  --learning_rate 5e-5 \
  --train_batch_size 8 \
  --eval_batch_size 32