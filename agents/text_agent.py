from openai import OpenAI
import json
class TextAgent:
    def __init__(self, client, model):
        self.client = client
        self.model = model
    def answer(self, query, context):
        prompt = open("prompts/text.txt").read()
        text_only_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt.format(query=query, context=context)}],
        ).choices[0].message.content

        return text_only_response
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