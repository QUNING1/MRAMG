from openai import OpenAI
import json
import re
class JudgeAgent:
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
        return self._detect_set_conflict(text_agent_response, visual_agent_response)
    def _detect_set_conflict(self, text_agent_response, visual_agent_response):
        """
        检测文本代理和视觉代理的响应是否存在选图集合冲突(P0)：宏观数量与范围分歧，该不该配图？全篇配几张图？
        """
        # 从text_agent_response中提取所有图片占位符
        list_text_images = self._extract_images(text_agent_response)
        # 从visual_agent_response中提取所有图片占位符
        list_visual_images = self._extract_images(visual_agent_response)
        
        # 检测 C1: 选图集合冲突 (Set Conflict)
        set_text_images = set(list_text_images)
        set_visual_images = set(list_visual_images)

        text_only = list(set_text_images - set_visual_images)
        visual_only = list(set_visual_images - set_text_images)

        has_set_conflict = bool(text_only or visual_only)

        conflicts = []
        if has_set_conflict:
            conflicts.append({
                "conflict_type": "set_conflict",
                "conflict_info": {
                    "images_only_in_text_agent_response": text_only,
                    "images_only_in_visual_agent_response": visual_only
                }
            })

        return conflicts
    def _extract_images(self, text):
        """
        使用正则表达式提取文本中的所有图片占位符，保持它们在文本中的原始顺序。
        支持格式如: <img1>, <img_12>, <img0> 等。
        """
        return re.findall(r'<img_?\d+>', text)
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