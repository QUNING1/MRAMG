import os
import json
import re
import argparse
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
from tqdm import tqdm

# --- 指标计算函数 ---

def get_image_metrics(gt_list, pred_list):
    gt_set, pred_set = set(gt_list), set(pred_list)
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def get_image_ordering_score(gt_list, pred_list, p1=3, p2=2, p3=1, p=3):
    n, m = len(gt_list), len(pred_list)
    if n == 0: return 0.0
    intersection_count = len(set(gt_list) & set(pred_list))
    recall_factor = intersection_count / n
    if recall_factor == 0: return 0.0

    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1): dp[i][0] = i * p1
    for j in range(m + 1): dp[0][j] = j * p2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt_list[i-1] == pred_list[j-1] else p3
            dp[i][j] = min(dp[i-1][j] + p1, dp[i][j-1] + p2, dp[i-1][j-1] + cost)
    
    dist_ab = dp[n][m]
    norm_dist = dist_ab / max(n, m)
    penalty = (1 / p) * min(norm_dist, p)
    return max(0.0, recall_factor * (1 - penalty))

def print_summary(file_name, metrics_list):
    """ 统一的打印格式 """
    if not metrics_list:
        return
    avg = np.mean(metrics_list, axis=0)
    print(f"\n[Summary] File: {file_name}")
    print(f"  Img Precision: {avg[0]:.4f} | Recall: {avg[1]:.4f} | F1: {avg[2]:.4f}")
    print(f"  Img Ordering:  {avg[3]:.4f}")
    print(f"  RougeLsum:     {avg[4]:.4f}")
    print(f"  BERTScore F1:  {avg[5]:.4f}")
    print("-" * 50)

# --- 主处理逻辑 ---

def process_file(file_path, args, r_scorer):
    file_name = os.path.basename(file_path)
    all_data = []
    
    # 1. 预读文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    if not all_data:
        return

    # 2. 检查是否已处理 (检查第一条数据是否有核心分数字段)
    if "bert_score_f1" in all_data[0]:
        print(f"File {file_name} already processed. Extracting existing scores...")
        existing_metrics = []
        for d in all_data:
            existing_metrics.append([
                d.get("image_precision", 0),
                d.get("image_recall", 0),
                d.get("image_f1", 0),
                d.get("image_ordering", 0),
                d.get("rougeLsum_f1", 0),
                d.get("bert_score_f1", 0)
            ])
        print_summary(file_name, existing_metrics)
        return

    # 3. 如果没处理过，开始计算
    print(f"Processing {file_name} (New calculations)...")
    preds_text = [d["final_output"] for d in all_data]
    gts_text = [d["ground_truth"] for d in all_data]

    # 批量计算 BERTScore 提高效率
    P, R, F1 = bert_score_func(
        preds_text, gts_text, 
        model_type=args.bert_path, 
        lang=args.lang, 
        device=args.device,
        verbose=False
    )
    
    new_metrics = []
    for i, data in enumerate(tqdm(all_data, desc=f"Calculating {file_name}")):
        # 图片指标提取与计算
        img_map = data.get("img_name_to_id", {})
        gt_imgs = [f"<img{img_map[img]}>" for img in data.get("images_list", [])]
        pred_imgs = re.findall(r"<img\d+>", data["final_output"])
        
        p, r, f1_img = get_image_metrics(gt_imgs, pred_imgs)
        ord_score = get_image_ordering_score(gt_imgs, pred_imgs)
        
        # Rouge-Lsum
        rouge_res = r_scorer.score(data["ground_truth"], data["final_output"])
        rl_f1 = rouge_res['rougeLsum'].fmeasure
        
        # 写入字典
        bs_f1 = F1[i].item()
        data.update({
            "image_precision": p,
            "image_recall": r,
            "image_f1": f1_img,
            "image_ordering": ord_score,
            "rougeLsum_f1": rl_f1,
            "bert_score_f1": bs_f1
        })
        new_metrics.append([p, r, f1_img, ord_score, rl_f1, bs_f1])

    # 4. 写回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # 5. 打印计算后的平均分
    print_summary(file_name, new_metrics)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Answer Evaluation Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .jsonl files")
    parser.add_argument("--bert_path", type=str, required=True, help="Local path to BERT model")
    parser.add_argument("--lang", type=str, default="en", help="Language for BERTScore")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    # 初始化一次 RougeScorer 提高效率
    r_scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)

    # 获取目录下所有 jsonl
    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')])
    
    if not files:
        print(f"No .jsonl files found in {args.input_dir}")
        return

    for filename in files:
        full_path = os.path.join(args.input_dir, filename)
        process_file(full_path, args, r_scorer)

if __name__ == "__main__":
    main()