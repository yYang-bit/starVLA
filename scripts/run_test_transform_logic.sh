#!/bin/bash
# 测试 ActionChunkTransform 逻辑（不加载真实数据）
# 输出日志到: logs/test_transform_logic.log

set -e

# 创建日志目录
mkdir -p logs

echo "========================================"
echo "测试 1: ActionChunkTransform 逻辑测试"
echo "========================================"
echo "开始时间: $(date)"
echo ""

# 激活环境并运行测试
source /mnt/home/liuyi/anaconda3/bin/activate starvla_qwen35
python scripts/test_action_chunk_transform_quick.py 2>&1 | tee logs/test_transform_logic.log

echo ""
echo "完成时间: $(date)"
echo "日志保存在: logs/test_transform_logic.log"
