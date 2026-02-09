#!/bin/bash
# 批量实验脚本
# 预计运行时间: 24-48 小时

set -e

cd "$(dirname "$0")"

LOG_DIR="/home/zzh/webpage-classification/data/results"
mkdir -p "$LOG_DIR"

STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S')
echo "=============================================="
echo " 批量实验开始: $STARTED_AT"
echo "=============================================="

run_exp() {
    local name="$1"
    shift
    echo ""
    echo "----------------------------------------------"
    echo " [$name] 开始: $(date '+%H:%M:%S')"
    echo "----------------------------------------------"
    if "$@"; then
        echo " [$name] 完成: $(date '+%H:%M:%S')"
    else
        echo " [$name] 失败 (exit=$?): $(date '+%H:%M:%S')"
    fi
}

# ==================== 多原型 ====================

run_exp "multi_proto_k3_topk2" python qwen_embedding_classifier_multiple_prototypes.py \
    --train 6000 --test 800 --n-prototypes 3 --top-k 2

run_exp "multi_proto_k4_topk2" python qwen_embedding_classifier_multiple_prototypes.py \
    --train 6000 --test 800 --n-prototypes 4 --top-k 2

run_exp "multi_proto_k5_topk2" python qwen_embedding_classifier_multiple_prototypes.py \
    --train 6000 --test 800 --n-prototypes 5 --top-k 2

run_exp "multi_proto_k6_topk2" python qwen_embedding_classifier_multiple_prototypes.py \
    --train 6000 --test 800 --n-prototypes 6 --top-k 2

run_exp "multi_proto_k8_topk2" python qwen_embedding_classifier_multiple_prototypes.py \
    --train 6000 --test 800 --n-prototypes 8 --top-k 2

# ==================== 对比学习 ====================

run_exp "contrastive_e10_l4_topk2" python qwen_embedding_classifier_contrastive_learning.py \
    --train 6000 --test 800 --epochs 10 --trainable-layers 4 --lr 1e-5 --top-k 2

run_exp "contrastive_e10_l4_lr5e6" python qwen_embedding_classifier_contrastive_learning.py \
    --train 6000 --test 800 --epochs 10 --trainable-layers 4 --lr 5e-6 --top-k 2

run_exp "contrastive_e15_l4_topk2" python qwen_embedding_classifier_contrastive_learning.py \
    --train 6000 --test 800 --epochs 15 --trainable-layers 4 --lr 1e-5 --top-k 2

run_exp "contrastive_e10_l6_topk2" python qwen_embedding_classifier_contrastive_learning.py \
    --train 6000 --test 800 --epochs 10 --trainable-layers 6 --lr 1e-5 --top-k 2

# ==================== 加权平均 ====================

run_exp "weighted_avg_iter2_topk2" python qwen_embedding_classifier_weighted_avg.py \
    --train 6000 --test 800 --iterations 2 --top-k 2

run_exp "weighted_avg_iter3_topk2" python qwen_embedding_classifier_weighted_avg.py \
    --train 6000 --test 800 --iterations 3 --top-k 2

run_exp "weighted_avg_iter4_topk2" python qwen_embedding_classifier_weighted_avg.py \
    --train 6000 --test 800 --iterations 4 --top-k 2

run_exp "weighted_avg_iter5_topk2" python qwen_embedding_classifier_weighted_avg.py \
    --train 6000 --test 800 --iterations 5 --top-k 2

# ==================== 汇总 ====================

echo ""
echo "=============================================="
echo " 全部实验完成!"
echo " 开始: $STARTED_AT"
echo " 结束: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
