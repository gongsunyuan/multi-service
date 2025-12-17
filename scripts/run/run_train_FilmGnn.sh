#!/bin/bash
taskset -c 0-63 \
msPython \
-m scripts.train.train_FilmGnn \
--yaml train_sdn.yaml \
| tee workspace/outputs/record/train_sdn_record_static_load