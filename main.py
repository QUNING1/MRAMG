import chromadb
import argparse
import json
from openai import OpenAI
import traceback
from agents import TextAgent, VisualAgent, JudgeAgent
from utils import build_prompt_from_chroma, parse_json
import argparse
import logging
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
logger = None

def run_single_debate(
    conflict_type,
    disputed_item, 
    defender, 
    challenger, 
    judge, 
    query, 
    context, 
    caption, 
    text_ans, 
    visual_ans, 
    all_img_paths,
):
    """
    单图辩论的 SOP 流程封装
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"🔍 冲突类型: {conflict_type}")
    logger.info(f"⚖️ 开启图片冲突庭审: {disputed_item}")
    logger.info(f"👉 举证方(Defender): {defender.role_name} | 质询方(Challenger): {challenger.role_name}")
    logger.info(f"{'='*50}")

    # ==========================================
    # Round 1: 举证方防守 (Defense)
    # ==========================================
    logger.info(f"🎙️ [Round 1] {defender.role_name} 正在陈述理由...")
    # 判断是否需要传图片（只有视觉模型看图，文本模型只看caption）
    defender_imgs = all_img_paths if defender.role_name == "Visual Agent" else None
    
    defense_argument = defender.defend(
        query=query, context=context, caption=caption,
        text_ans=text_ans, visual_ans=visual_ans,
        disputed_item=disputed_item, challenger_role=challenger.role_name,
        img_paths=defender_imgs, conflict_type=conflict_type,
    )
    logger.info(f"   [辩词] {defense_argument.strip()}")

    # ==========================================
    # Round 2: 质询方攻击 (Critique)
    # ==========================================
    logger.info(f"\n🔍 [Round 2] {challenger.role_name} 正在审查并质询...")
    challenger_imgs = all_img_paths if challenger.role_name == "Visual Agent" else None
    
    critique_response = challenger.critique(
        query=query, context=context, caption=caption,
        text_ans=text_ans, visual_ans=visual_ans,
        disputed_item=disputed_item, defender_role=defender.role_name,
        defender_argument=defense_argument,
        img_paths=challenger_imgs, conflict_type=conflict_type, 
    )
    
    # 尝试解析 JSON 结果
    try:
        critique_data = parse_json(critique_response)
        decision = critique_data.get("decision", "REJECT").upper()
        reasoning = critique_data.get("reasoning", "")
    except Exception as e:
        logger.error(f"   [解析质询JSON失败]: {e}. 原始返回: {critique_response}")
        decision = "REJECT" # 只要解析失败，强制视为拒不妥协，交由Judge裁决
        reasoning = critique_response
        
    logger.info(f"   [判决] {decision}")
    logger.info(f"   [理由] {reasoning.strip()}")

    if decision == "ACCEPT":
        logger.info(f"\n🤝 [庭审结束] 双方达成共识，{challenger.role_name} 接受了提案。无需 Judge 介入。")
        
        # 动态决定返回结果 (Resolution)
        if conflict_type == "set_conflict":
            final_resolution = "INCLUDE"
        elif conflict_type == "order_conflict":
            # 如果是顺序冲突，胜利果实就是防守方（Defender）主张的数组
            defender_key = f"{defender.role_name.lower().replace(' ', '_')}_img_order"
            final_resolution = disputed_item.get(defender_key)
            
        return {
            "winner": defender.role_name,  # 既然挑战方接受了，赢家就是防守方
            "resolution": final_resolution, # 返回具体的动作或数组
            "reasoning": reasoning
        }
    # ==========================================
    # Round 3: 法官裁决 (Judge Verdict)
    # ==========================================
    logger.info(f"\n🔨 [Round 3] 双方拒绝妥协，Chief Judge 介入最终裁定...")
    # Judge 是上帝视角，必须传全套图片
    judge_verdict_str = judge.judge(
        query=query, 
        # img_paths=all_img_paths, 
        context=context, caption=caption,
        defender_role=defender.role_name, challenger_role=challenger.role_name,
        defender_argument=defense_argument, challenger_argument=reasoning,
        disputed_item=disputed_item, conflict_type=conflict_type,
    )
    
    try:
        judge_data = parse_json(judge_verdict_str)
        logger.info(f"   [最终胜方] {judge_data.get('winner')}")
        logger.info(f"   [处置结果] {judge_data.get('resolution')} {disputed_item}")
        logger.info(f"   [法官判词] {judge_data.get('verdict_reasoning')}")
        return judge_data
    except Exception as e:
        logger.error(f"   [解析JudgeJSON失败]: {e}. 原始返回: {judge_verdict_str}")
        return {"winner": "UNKNOWN", "resolution": "UNKNOWN", "reasoning": judge_verdict_str}


def sanitize_filename(name):
    """清理模型名称中的特殊字符，方便用作文件名"""
    return name.replace("/", "-").replace(":", "-").replace(".", "_")

def parse_args():
    parser = argparse.ArgumentParser(description="MRAMG Benchmark Testing Script")
    parser.add_argument("--doc_name", type=str, required=True, help="Document name (e.g., manual, arxiv)")
    parser.add_argument("--text_model", type=str, default="gpt-4o", help="Model for Text Agent")
    parser.add_argument("--visual_model", type=str, default="gpt-4o", help="Model for Visual Agent")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model for Judge Agent")
    parser.add_argument("--input_dir", type=str, default="test", help="Directory containing the input jsonl files")
    parser.add_argument("--output_dir", type=str, default="test/outputs", help="Directory to save the generated jsonl files")
    parser.add_argument("--top_k", type=int, default=10, help="Number of chunks to retrieve from ChromaDB")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API Key")
    parser.add_argument("--base_url", type=str, default="https://api.qingyuntop.top/v1", help="OpenAI API Base URL")
    parser.add_argument("--num_workers", type=int, default=5, help="并发执行的线程数 (建议 4-10 之间)")

    return parser.parse_args()

def process_single_question(question, question_emb, doc_name, collection, text_agent, visual_agent, judge_agent, top_k):
    """封装单条数据的完整处理流水线"""

    chunks = collection.query(
        query_embeddings=[question_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"] 
    )
    
    content, caption, all_img_paths = build_prompt_from_chroma(doc_name, chunks)
    logger.info(f"检索完成 | 命中 chunks: {len(chunks['documents'][0])} | 图片数量: {len(all_img_paths)}")
    
    # 2. 独立生成双轨草稿
    text_agent_response = text_agent.answer(question, content, caption)
    visual_agent_response = visual_agent.answer(question, all_img_paths, content, caption)

    # 3. 静态冲突检测
    conflicts = judge_agent.detect_conflict(text_agent_response, visual_agent_response)
    logger.info(f"冲突检测完成，共发现 {len(conflicts)} 种类型的冲突: {conflicts}")

    # 全局辩论账本
    debate_ledger = []

    # 4. 逐个化解冲突
    for conflict in conflicts:
        if conflict["conflict_type"] == "set_conflict":
            info = conflict["conflict_info"]
            text_only_imgs = info.get("images_only_in_text_agent_response", [])
            visual_only_imgs = info.get("images_only_in_visual_agent_response", [])
            
            # --- 场景 A: 文本智能体要求的图 (Text_Only) ---
            for img in text_only_imgs:
                res = run_single_debate(
                    conflict_type="set_conflict",
                    disputed_item={"disputed_image": img},
                    defender=text_agent,
                    challenger=visual_agent,
                    judge=judge_agent,
                    query=question, context=content, caption=caption,
                    text_ans=text_agent_response, visual_ans=visual_agent_response,
                    all_img_paths=all_img_paths
                )
                debate_ledger.append({
                    "conflict_type": "set_conflict",
                    "target_image": img,
                    "resolution": res
                })

            # --- 场景 B: 视觉智能体强加的图 (Visual_Only) ---
            for img in visual_only_imgs:
                res = run_single_debate(
                    conflict_type="set_conflict",
                    disputed_item={"disputed_image": img},
                    defender=visual_agent,
                    challenger=text_agent,
                    judge=judge_agent,
                    query=question, context=content, caption=caption,
                    text_ans=text_agent_response, visual_ans=visual_agent_response,
                    all_img_paths=all_img_paths
                )
                debate_ledger.append({
                    "conflict_type": "set_conflict",
                    "target_image": img,
                    "resolution": res
                })

        elif conflict["conflict_type"] == "order_conflict":
            info = conflict["conflict_info"]
            text_order = info.get("text_order", [])
            visual_order = info.get("visual_order", [])
            
            if not text_order or not visual_order: 
                logger.warning(f"顺序冲突检测到空顺序: Text Order: {text_order}, Visual Order: {visual_order}")
                continue  
            
            disputed_order_str = f"Text Agent Order: {text_order} vs Visual Agent Order: {visual_order}"
            disputed_item = {
                "text_agent_img_order": text_order,
                "visual_agent_img_order": visual_order,
            }
            
            res = run_single_debate(
                conflict_type="order_conflict",
                disputed_item=disputed_item, 
                defender=text_agent,
                challenger=visual_agent,
                judge=judge_agent,
                query=question, context=content, caption=caption,
                text_ans=text_agent_response, visual_ans=visual_agent_response,
                all_img_paths=all_img_paths
            )
            debate_ledger.append({
                "conflict_type": "order_conflict",
                "target_sequence": disputed_order_str,
                "resolution": res
            })

    # 5. 终局排版
    final_output = judge_agent.synthesize(
        query=question,
        text_draft=text_agent_response, 
        # visual_draft=visual_agent_response, # 要不要加visual draft，需要讨论
        debate_ledger=debate_ledger,
        context=content, 
        caption=caption,
    )

    return text_agent_response, visual_agent_response, conflicts, debate_ledger, final_output

def main():
    args = parse_args()

    # 初始化 ChromaDB
    chromadb_client = chromadb.PersistentClient(path="/data2/qn/MRAMG/chromadb")
    collection = chromadb_client.get_or_create_collection(name=f"doc_{args.doc_name}")
    
    # 初始化 OpenAI Client (复用)
    client = OpenAI(
        api_key=args.api_key, 
        base_url=args.base_url, 
    )
    
    # 实例化所有 Agents (根据传入的参数配置模型)
    text_agent = TextAgent(client, model=args.text_model)
    visual_agent = VisualAgent(client, model=args.visual_model)
    judge_agent = JudgeAgent(client, model=args.judge_model)

    # 配置文件路径
    input_filepath = os.path.join(args.input_dir, f"{args.doc_name}_mqa.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构造输出文件名: doc_name_T-model_V-model_J-model.jsonl
    safe_t_model = sanitize_filename(args.text_model)
    safe_v_model = sanitize_filename(args.visual_model)
    safe_j_model = sanitize_filename(args.judge_model)
    output_filename = f"{args.doc_name}_T-{safe_t_model}_V-{safe_v_model}_J-{safe_j_model}.jsonl"
    output_filepath = os.path.join(args.output_dir, output_filename)

    # ===== 初始化日志 =====
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = output_filename.replace(".jsonl", ".log")
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filepath, encoding="utf-8")
        ]
    )

    global logger
    logger = logging.getLogger(__name__)

    logger.info(f"日志文件: {log_filepath}")

    if not os.path.exists(input_filepath):
        logger.error(f"❌ 找不到输入文件: {input_filepath}")
        return

    logger.info(f"🚀 开始批量处理! 输入: {input_filepath} | 输出: {output_filepath}")

    # 读取并逐行处理
    processed_count = 0
    error_count = 0
    
    with open(input_filepath, 'r', encoding='utf-8') as f_in, \
         open(output_filepath, 'w', encoding='utf-8') as f_out:
        
        lines = [(idx, line) for idx, line in enumerate(f_in) if line.strip()]

        def worker(line_idx, line):
            data = json.loads(line)
            question = data.get('question', '')
            question_emb = data.get('query_emb', None)

            logger.info(f"\n=======================================================")
            logger.info(f"📝 正在处理第 {line_idx + 1} 条数据 | Question: {question}")

            if question_emb is None:
                logger.warning(f"⚠ 第 {line_idx + 1} 条数据没有 query_emb，跳过推理")

                data["text_agent_response"] = None
                data["visual_agent_response"] = None
                data["conflicts"] = []
                data["debate_ledger"] = []
                data["final_output"] = None

                return line_idx, data

            text_ans, vis_ans, conflicts, ledger, final_ans = process_single_question(
                question=question,
                question_emb=question_emb,
                doc_name=args.doc_name,
                collection=collection,
                text_agent=text_agent,
                visual_agent=visual_agent,
                judge_agent=judge_agent,
                top_k=args.top_k
            )

            if question_emb is not None:
                del data["query_emb"]

            data["text_agent_response"] = text_ans
            data["visual_agent_response"] = vis_ans
            data["conflicts"] = conflicts
            data["debate_ledger"] = ledger
            data["final_output"] = final_ans

            return line_idx, data

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:

            futures = [
                executor.submit(worker, idx, line)
                for idx, line in lines
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):

                try:
                    line_idx, data = future.result()

                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    f_out.flush()

                    processed_count += 1
                    logger.info(f"✅ 第 {line_idx + 1} 条数据处理成功并保存。")

                except Exception as e:
                    error_count += 1
                    logger.error(f"❌ 数据处理失败! 错误信息: {e}")
                    logger.error(traceback.format_exc())

    logger.info(f"\n🎉 评测任务全部完成！")
    logger.info(f"📊 统计: 成功处理 {processed_count} 条，失败 {error_count} 条。")
    logger.info(f"💾 结果已保存至: {output_filepath}")

if __name__ == "__main__":
    main()