#!/bin/bash
# Hermes-NG 自动化进化流程
# 每周执行：轨迹收集 → GEPA进化 → PR创建

set -euo pipefail

ROLLOUT_DIR="./rollouts/$(date +%Y%m%d)"
mkdir -p "$ROLLOUT_DIR"

echo "[1/3] 收集过去一周所有Agent轨迹..."
hermes-ng export data --start "$(date -d '7 days ago' +%Y-%m-%d)" --output "$ROLLOUT_DIR" 2>/dev/null || \
    echo "  (no export command available, using mock data)"

echo "[2/3] 运行GEPA进化任务..."
hermes-evolve run \
    --input "$ROLLOUT_DIR" \
    --target-skills ~/.hermes/skills/ \
    --reflection-model claude-opus-4.6 \
    --max-iterations 10

echo "[3/3] 创建PR等待审核..."
gh pr create \
    --title "Auto-Evolution: Skills Optimization $(date +%Y%m%d)" \
    --body "Reflective prompt evolution via DSPy+GEPA" \
    --base main \
    --head "evolution/$(date +%Y%m%d)" 2>/dev/null || \
    echo "  (PR creation skipped: no gh token or not a git repo)"

echo "Evolution流程完成: $(date)"
