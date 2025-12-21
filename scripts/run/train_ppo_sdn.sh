#!/bin/bash
taskset -c 0-63 \
msPython \
-m scripts.train.ppo_train \
--yaml ppo_train.yaml \
--checkpoint workspace/checkpoints/ospf_train/20251219_195558_ospf_train/final_model.pth