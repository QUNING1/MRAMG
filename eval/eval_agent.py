from agents import BaseAgent
from openai import OpenAI
class EvalAgent(BaseAgent):
    def __init__(self, client: OpenAI, model: str):
        super().__init__(client, model, role_name="Eval Agent")
    def get_score_from_response(self, response: str) -> float:
        pattern = r"<([^>]+)>(\d+\.?\d*)</\1>"
    
        # 查找所有匹配项
        matches = re.findall(pattern, response)
        if matches:
            # 提取第一个匹配到的分数（通常只有一个评分标签）
            score_str = matches[0][1]
            try:
                # 转换为float类型
                score = float(score_str)
                return score
            except ValueError:
                print(f"警告：提取的内容'{score_str}'无法转换为浮点数")
                return 0.0
        else:
            print("错误：未找到<标签>分数</标签>格式的评分内容")
            return 0.0

    def eval(self, prompt_template, query, context, caption, answer):
        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption,
            answer=answer,
        )

        content = self._build_content(formatted_prompt, img_paths=None)
        
        response = self._call_llm(content, temperature=0.0)
        return self.get_score_from_response(response)
