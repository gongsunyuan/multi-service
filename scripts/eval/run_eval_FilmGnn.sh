#! /bin/bash
taskset -c 0-63 \
msPython \
-m scripts.eval.eval_FilmGnn \
--yaml eval_sdn.yaml \
--checkpoint workspace/checkpoints/sdn/sdn_exp_20251215183613/finalmodel.pth \
--load_flow 60 \
| tee workspace/outputs/record/eval_sdn_record_load60

taskset -c 0-63 \
msPython \
-m scripts.eval.eval_FilmGnn \
--yaml eval_sdn.yaml \
--checkpoint workspace/checkpoints/sdn/sdn_exp_20251215183613/finalmodel.pth \
--load_flow 90 \
| tee workspace/outputs/record/eval_sdn_record_load90

taskset -c 0-63 \
msPython \
-m scripts.eval.eval_FilmGnn \
--yaml eval_sdn.yaml \
--checkpoint workspace/checkpoints/sdn/sdn_exp_20251215183613/finalmodel.pth \
--load_flow 150 \
| tee workspace/outputs/record/eval_sdn_record_load150

taskset -c 0-63 \
msPython \
-m scripts.eval.eval_FilmGnn \
--yaml eval_sdn.yaml \
--checkpoint workspace/checkpoints/sdn/sdn_exp_20251215183613/finalmodel.pth \
--load_flow 200 \
| tee workspace/outputs/record/eval_sdn_record_load200