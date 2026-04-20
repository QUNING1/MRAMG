## 快速开始

### 环境配置

```bash
# 创建并激活 conda 环境
conda create -n mramg python=3.11
conda activate mramg

# 安装依赖
pip install -r requirements.txt
```

### 数据&模型下载

#### 1. MRAMG-Bench 数据集

MRAMG-Bench 数据集包含多模态问答数据和图片资源。cite[https://huggingface.co/datasets/MRAMG/MRAMG-Bench/tree/main]
下载到当前目录下


#### 2. 嵌入模型 (Embedding Model)

为question和chunks生成embedding。cite[https://huggingface.co/BAAI/bge-m3]

#### 3. BERT 评估模型

评估阶段使用 RoBERTa-Large 作为 BERTScore 模型。cite[https://huggingface.co/FacebookAI/roberta-large]

### chromadb 数据库创建

见bmk_process.ipynb

### 数据集处理

通过emb_loads.py脚本处理数据集，将question的emb一次性落好（放到MRAMG-Bench/mqa_with_emb文件夹下），避免每次评估都重新计算。

### 启动脚本
```
sh run.sh
```

### 评估脚本
```
sh eval.sh
```


