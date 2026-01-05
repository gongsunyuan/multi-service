#!/bin/bash
taskset -c 0-63 \
msPython \
-m scripts.train.ppo \
--yaml ppo.yaml \