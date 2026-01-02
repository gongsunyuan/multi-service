#!/bin/bash
taskset -c 0-63 \
msPython \
-m scripts.train.ppo_train \
--yaml ppo_train.yaml \