#!/bin/bash

# ==========================================
# MRAMG Benchmark 多智能体测试启动脚本
# ==========================================

# 1. 设置 API 信息 (替换为你自己的真实 Key)
export API_KEY="sk-NAKH2KjEcrfJyRdUxa5Ck52KVXRIJ1K6m5wuOIN6jXGizxg1"
export BASE_URL="https://api.qingyuntop.top/v1"

# 2. 设置评测文档与模型配置
DOC_NAMES=("arxiv" "manual" "recipe" "web" "wiki" "wit")                  # e.g., manual, arxiv
# DOC_NAMES=("recipe" "web" "wiki" "wit")
TEXT_MODEL="gpt-4o"
VISUAL_MODEL="gpt-4o"
JUDGE_MODEL="gpt-4o"

# 3. 设置工程路径与并发度
INPUT_DIR="MRAMG-Bench/mqa_with_emb"
OUTPUT_DIR="outputs"
TOP_K=10
NUM_WORKERS=5

# 4. 遍历文档列表执行测试
for DOC_NAME in "${DOC_NAMES[@]}"; do
    # 打印启动信息
    echo "========================================================"
    echo "🚀 启动 MRAMG Benchmark 评测..."
    echo "📄 当前处理文档: ${DOC_NAME}"
    echo "🤖 模型配置 (Text/Visual/Judge): ${TEXT_MODEL} / ${VISUAL_MODEL} / ${JUDGE_MODEL}"
    echo "⚡ 并发数: ${NUM_WORKERS}"
    echo "--------------------------------------------------------"

    # 执行 Python 脚本
    python main.py \
        --doc_name "${DOC_NAME}" \
        --text_model "${TEXT_MODEL}" \
        --visual_model "${VISUAL_MODEL}" \
        --judge_model "${JUDGE_MODEL}" \
        --input_dir "${INPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --top_k ${TOP_K} \
        --api_key "${API_KEY}" \
        --base_url "${BASE_URL}" \
        # --num_workers ${NUM_WORKERS}

    echo "--------------------------------------------------------"
    echo "✅ 文档 ${DOC_NAME} 运行结束。"
    echo ""
done

echo "🎉 所有文档的评测任务均已完成！"