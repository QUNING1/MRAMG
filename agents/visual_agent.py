from openai import OpenAI
from .base_agent import BaseAgent
class VisualAgent(BaseAgent):
    def __init__(self, client: OpenAI, model: str, img_server_port: int, model_mode: str):
        super().__init__(client, model, role_name="Visual Agent", img_server_port=img_server_port, model_mode=model_mode)
        
    def answer(self, query, img_paths, context, caption):
        """生成初始图文排版草稿 (基于视觉特征)"""
        prompt_template = open("prompts/visual.txt").read()
        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption
        )
        
        # Visual Agent 初始生成时必须传入所有 img_paths
        content = self._build_content(formatted_prompt, img_paths=img_paths)
        
        return self._call_llm(content, temperature=0.0)
    
if __name__ == "__main__":
    client = OpenAI(
        api_key="llama", 
        base_url="http://127.0.0.1:8005/v1", 
    )
    agent = VisualAgent(client, model="/data2/qn/KGQA/models/Qwen2.5-VL-72B-Instruct", img_server_port=8009, model_mode="vllm")
    query = "图中描述的是什么内容?"
    img_paths = ["MRAMG-Bench/IMAGE/IMAGE/images/ARXIV/2403_14627v2_1.png", "MRAMG-Bench/IMAGE/IMAGE/images/ARXIV/2403_14627v2_1.png"]
    context = " "
    caption = "图"

    strategy = agent.answer(query, img_paths, context, caption)
    print(strategy)