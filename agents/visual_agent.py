import sys
import os

from openai import OpenAI
import json
import base64
from pathlib import Path
class VisualAgent:
    def __init__(self, client: OpenAI, model: str):
        super().__init__(client, model, role_name="Visual Agent")
        
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
        api_key="sk-NAKH2KjEcrfJyRdUxa5Ck52KVXRIJ1K6m5wuOIN6jXGizxg1", 
        base_url="https://api.qingyuntop.top/v1", 
    )
    agent = VisualAgent(client, model="gpt-4o")
    query = "图中描述的是什么内容?"
    img_paths = ["MRAMG-Bench/IMAGE/IMAGE/images/ARXIV/2403_14627v2_1.png"]
    context = ""
    caption = "图"

    strategy = agent.answer(query, img_paths, context, caption)
    print(strategy)