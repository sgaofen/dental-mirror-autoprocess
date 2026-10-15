#!/bin/bash
#SBATCH -J dental
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o logs/%x_%j.out          # 相对提交目录；提交前先 mkdir -p logs
# 牙齿关键点对照实验（一组一张卡）。数据先拷到节点本地盘再解压，避免 BeeGFS 卡顿拖慢读图。
set -eo pipefail
D=${D:?请先 export D=集群上放数据和脚本的目录}
export YOLO_CONFIG_DIR=$D/.ultra_cfg; mkdir -p "$YOLO_CONFIG_DIR"
LOCAL=${TMPDIR:-/tmp}/dental_$SLURM_JOB_ID; mkdir -p "$LOCAL"
for d in $DATA ${COMMON:-dataset_C}; do [ -d "$LOCAL/$d" ] || tar -xf "$D/$d.tar" -C "$LOCAL"; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ARM=$ARM DATA=$DATA MODEL=$MODEL EPOCHS=${EPOCHS:-40} FLIPLR=${FLIPLR:-0.5} IMGSZ=${IMGSZ:-320} COMMON=${COMMON:-dataset_C} DROOT=$LOCAL OUTD=$D \
  $D/venv/bin/python -u $D/${SCRIPT:-train_arm.py} 2>&1 | tr '\r' '\n' | grep --line-buffered -vE '^\s*[0-9]+/[0-9]+ .*(it/s|s/it)' || true
grep -q DONE <(tail -5 $SLURM_SUBMIT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out) || { echo "★ 训练没有正常结束"; exit 1; }
rm -rf "$LOCAL"
