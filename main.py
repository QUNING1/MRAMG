import chromadb
import argparse
from openai import OpenAI
from agents import TextAgent, VisualAgent
from sentence_transformers import SentenceTransformer
from utils import build_prompt_from_chroma

# logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

emb_model = SentenceTransformer("/data2/qn/MRAMG/models/bge-m3")

def main():
    doc_name = "arxiv"
    chromadb_client = chromadb.PersistentClient(path="/data2/qn/MRAMG/chromadb")
    collection = chromadb_client.get_or_create_collection(name=f"doc_{doc_name}")
    question = "How does MVSplat achieve efficient 3D Gaussian prediction and fast inference?"

    query_emb = emb_model.encode([question]).tolist()

    chunks = collection.query(
        query_embeddings=query_emb,
        n_results=3,
        # 显式指定需要包含的内容：默认只有 metadatas 和 documents
        include=["documents", "metadatas", "distances"] 
    )
    
    content, caption, all_img_paths = build_prompt_from_chroma(doc_name, chunks)
    
    logger.info(f"正在回答问题: {question}，检索到chunks nums: {len(chunks['documents'][0])}， 图片nums: {len(all_img_paths)}")
    # import pdb; pdb.set_trace()
    client = OpenAI(
        api_key="sk-NAKH2KjEcrfJyRdUxa5Ck52KVXRIJ1K6m5wuOIN6jXGizxg1", 
        base_url="https://api.qingyuntop.top/v1", 
    )
    text_agent = TextAgent(client, model="gpt-4o")
    visual_agent = VisualAgent(client, model="gpt-4o")


    text_response = text_agent.answer(question, content, caption)
    logger.info(f"调用text_agent, 得到text_response: {text_response}")

    visual_response = visual_agent.answer(question, all_img_paths, content, caption)
    logger.info(f"调用visual_agent, 得到visual_response: {visual_response}")
if __name__ == "__main__":
    main()
# proposals = [propose(llm, query) for _ in range(3)]

# votes = []
# for agent in [text_agent, visual_agent, judge_agent]:
#     votes.append(agent.vote(proposals))

# selected_proposal = aggregate_votes(votes)

# text_answer = text_agent.answer(query, selected_proposal)
# visual_answer = visual_agent.answer(query, selected_proposal)