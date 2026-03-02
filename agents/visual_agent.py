import sys
import os

from openai import OpenAI
import json
import base64
from pathlib import Path
class VisualAgent:
    def __init__(self, client: OpenAI, model: str):
        if client is None:
            raise ValueError("OpenAI client must be provided")
        self.client = client
        self.model = model

    def _encode_image(self, img_path: str) -> str:
        """
        将本地图片转为 base64 data URL
        """
        img_path = Path(img_path)

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        suffix = img_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif suffix == ".png":
            mime = "image/png"
        elif suffix == ".webp":
            mime = "image/webp"
        else:
            raise ValueError(f"Unsupported image format: {suffix}")

        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime};base64,{encoded}"

    def answer(self, query, img_paths, context, caption):
        prompt_template = open("prompts/visual.txt").read()
        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption
        )
        # 构造多模态输入
        content = []

        # 主 prompt
        content.append({
            "type": "input_text",
            "text": formatted_prompt
        })

        # 图片
        for img_path in img_paths:
            encoded_img = self._encode_image(img_path)

            content.append({
                "type": "input_image",
                "image_url": encoded_img
            })

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0
        )

        return response.output_text
    # 辩论
    # 怎么在辩论中自动填充text_only_response
    def debate(self, query, context, text_only_response, claim, opponent_view):
        prompt = open("prompts/debate.txt").read()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt.format(query=query, context=context, text_only_response=text_only_response, claim=claim, opponent_view=opponent_view)}],
        ).choices[0].message.content

        return response
    
    def parse_debate(self, response):
        # 解析LLM输出的辩论字符串为JSON格式,若解析失败则尝试用正则表达式提取
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            match = re.search(r'"claim":\s*"([^"]+)"', response)
            if match:
                return {"claim": match.group(1)}
            return None
        
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