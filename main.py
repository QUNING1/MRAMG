import chromadb
import argparse
import json
from openai import OpenAI
from agents import TextAgent, VisualAgent, JudgeAgent
from sentence_transformers import SentenceTransformer
from utils import build_prompt_from_chroma

# logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

emb_model = SentenceTransformer("/data2/qn/MRAMG/models/bge-m3")


def run_single_debate(disputed_image, defender, challenger, judge, query, context, caption, text_ans, visual_ans, all_img_paths):
    """
    单图辩论的 SOP 流程封装
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"⚖️ 开启图片冲突庭审: {disputed_image}")
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
        disputed_image=disputed_image, challenger_role=challenger.role_name,
        img_paths=defender_imgs
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
        disputed_image=disputed_image, defender_role=defender.role_name,
        defender_argument=defense_argument,
        img_paths=challenger_imgs
    )
    
    # 尝试解析 JSON 结果
    try:
        # 清洗可能存在的 Markdown 代码块标记 (```json ... ```)
        clean_json_str = critique_response.strip("`").removeprefix("json").strip()
        critique_data = json.loads(clean_json_str)
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
        return {"winner": "BOTH", "action": "INCLUDE", "reasoning": reasoning}

    # ==========================================
    # Round 3: 法官裁决 (Judge Verdict)
    # ==========================================
    logger.info(f"\n🔨 [Round 3] 双方拒绝妥协，Chief Judge 介入最终裁定...")
    # Judge 是上帝视角，必须传全套图片
    judge_verdict_str = judge.answer(
        query=query, img_paths=all_img_paths, context=context, caption=caption,
        defender_role=defender.role_name, challenger_role=challenger.role_name,
        defender_argument=defense_argument, challenger_argument=reasoning,
        disputed_image=disputed_image
    )
    
    try:
        clean_verdict_str = judge_verdict_str.strip("`").removeprefix("json").strip()
        judge_data = json.loads(clean_verdict_str)
        logger.info(f"   [最终胜方] {judge_data.get('winner')}")
        logger.info(f"   [处置结果] {judge_data.get('action')} {disputed_image}")
        logger.info(f"   [法官判词] {judge_data.get('verdict_reasoning')}")
        return judge_data
    except Exception as e:
        logger.error(f"   [解析JudgeJSON失败]: {e}. 原始返回: {judge_verdict_str}")
        return {"winner": "UNKNOWN", "action": "UNKNOWN", "reasoning": judge_verdict_str}


def main():
    doc_name = "manual"
    chromadb_client = chromadb.PersistentClient(path="/data2/qn/MRAMG/chromadb")
    collection = chromadb_client.get_or_create_collection(name=f"doc_{doc_name}")
    question = "How can you install the WIDCOMM bluetooth stack  for a bluetooth laser mouse?"

    query_emb = emb_model.encode([question]).tolist()

    chunks = collection.query(
        query_embeddings=query_emb,
        n_results=3,
        include=["documents", "metadatas", "distances"] 
    )
    
    content, caption, all_img_paths = build_prompt_from_chroma(doc_name, chunks)
    
    logger.info(f"正在回答问题: {question}，检索到chunks nums: {len(chunks['documents'][0])}， 图片nums: {len(all_img_paths)}")
    
    client = OpenAI(
        api_key="sk-NAKH2KjEcrfJyRdUxa5Ck52KVXRIJ1K6m5wuOIN6jXGizxg1", 
        base_url="https://api.qingyuntop.top/v1", 
    )
    
    # 实例化所有 Agents
    text_agent = TextAgent(client, model="gpt-4o")
    visual_agent = VisualAgent(client, model="gpt-4o")
    judge_agent = JudgeAgent(client, model="gpt-4o")

    # 1. 独立生成双轨草稿
    text_agent_response = text_agent.answer(question, content, caption)
    logger.info(f"调用text_agent, 得到text_response:\n{text_agent_response}")

    visual_agent_response = visual_agent.answer(question, all_img_paths, content, caption)
    logger.info(f"调用visual_agent, 得到visual_response:\n{visual_agent_response}")

    import pdb; pdb.set_trace()
    # 2. 冲突检测
    conflicts = judge_agent.detect_conflict(text_agent_response, visual_agent_response)
    logger.info(f"冲突检测完成，共发现 {len(conflicts)} 种类型的冲突: {conflicts}")

    # 3. 逐个化解冲突 (Orchestration)
    for conflict in conflicts:
        if conflict["conflict_type"] == "set_conflict":
            info = conflict["conflict_info"]
            
            text_only_imgs = info.get("images_only_in_text_agent_response", [])
            visual_only_imgs = info.get("images_only_in_visual_agent_response", [])
            
            # --- 场景 A: 文本智能体要求的图 (Text_Only) ---
            # 文本主张，视觉质询
            for img in text_only_imgs:
                run_single_debate(
                    disputed_image=img,
                    defender=text_agent,      # Text 先手防守
                    challenger=visual_agent,  # Visual 质询
                    judge=judge_agent,
                    query=question, context=content, caption=caption,
                    text_ans=text_agent_response, visual_ans=visual_agent_response,
                    all_img_paths=all_img_paths
                )

            # --- 场景 B: 视觉智能体强加的图 (Visual_Only) ---
            # 视觉主张，文本质询
            for img in visual_only_imgs:
                run_single_debate(
                    disputed_image=img,
                    defender=visual_agent,    # Visual 先手防守
                    challenger=text_agent,    # Text 质询
                    judge=judge_agent,
                    query=question, context=content, caption=caption,
                    text_ans=text_agent_response, visual_ans=visual_agent_response,
                    all_img_paths=all_img_paths
                )

    logger.info("\n🎉 所有图像冲突已处理完毕。")

if __name__ == "__main__":
    main()