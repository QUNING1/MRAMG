from openai import OpenAI
from .base_agent import BaseAgent
class TextAgent(BaseAgent):
    def __init__(self, client: OpenAI, model: str):
        super().__init__(client, model, role_name="Text Agent")

    def answer(self, query, context, caption):
        prompt_template = open("prompts/text.txt").read()

        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption
        )

        # Text Agent 不需要传 img_paths
        content = self._build_content(formatted_prompt, img_paths=None)
        
        # 初始草稿要求严谨，temperature=0
        return self._call_llm(content, temperature=0.0)
        
if __name__ == "__main__":
    client = OpenAI(
        api_key="sk-NAKH2KjEcrfJyRdUxa5Ck52KVXRIJ1K6m5wuOIN6jXGizxg1", 
        base_url="https://api.qingyuntop.top/v1", 
    )
    agent = TextAgent(client, model="gpt-5")
    query = "如何做西红柿炒鸡蛋?"
    strategy = agent.answer(query, "", "")
    print(strategy)