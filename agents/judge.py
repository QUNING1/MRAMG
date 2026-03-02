from openai import OpenAI
import json
class TextAgent:
    def __init__(self, client: OpenAI, model: str):
        if client is None:
            raise ValueError("OpenAI client must be provided")

        self.client = client
        self.model = model

    def answer(self, query, context, caption):
        prompt_template = open("prompts/text.txt").read()

        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": formatted_prompt
                        }
                    ]
                }
            ],
            temperature=0
        )

        return response.output_text
    # 检测冲突
    def detect_conflict(self, text_agent_response, visual_agent_response):
        """
        检测文本代理和视觉代理的响应是否存在冲突。
        总共包含4种类型冲突：
        1. 选图集合冲突(P0)：宏观数量与范围分歧，该不该配图？全篇配几张图？
        2. 插入位置冲突(P2): 同图不同点，选了同一张图，但挂载的知识点/逻辑锚点不同。
        3. 局部物证冲突(P2)：同点不同图，在同一个知识点上，各自提交了不同的视觉证据。
        4. 顺序与时序冲突(P1)：因果倒置，图片的整体排列顺序违背了物理常识或时间线。
        """
        
        prompt_template = open("prompts/conflict_detection.txt").read()

        formatted_prompt = prompt_template.format(
            text_agent_response=text_agent_response,
            visual_agent_response=visual_agent_response
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": formatted_prompt
                        }
                    ]
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
    agent = TextAgent(client, model="gpt-5")
    query = "如何做西红柿炒鸡蛋?"
    strategy = agent.answer(query, "")
    print(strategy)