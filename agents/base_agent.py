from openai import OpenAI
import json
import base64
from pathlib import Path
class BaseAgent:
    def __init__(self, client: OpenAI, model: str, role_name: str):
        if client is None:
            raise ValueError("OpenAI client must be provided")

        self.client = client
        self.model = model
        self.role_name = role_name

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

    def _build_content(self, text_prompt: str, img_paths: list = None) -> list:
        """
        通用多模态 payload 构造器。
        如果有 img_paths，就自动拼接图片；如果没有，就是纯文本。
        """
        content = [{"type": "input_text", "text": text_prompt}]
        
        if img_paths:
            for img_path in img_paths:
                encoded_img = self._encode_image(img_path)
                content.append({
                    "type": "input_image",
                    "image_url": encoded_img
                })
        return content
    
    def _call_llm(self, content: list, temperature: float = 0.3):
        """
        统一封装 LLM 调用逻辑
        """
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=temperature
        )
        return response.output_text
    
    # --- 辩论核心方法 ---
    def defend(self, query, context, caption, text_ans, visual_ans, disputed_image, challenger_role, img_paths=None):
        """举证方法"""
        prompt_template = open("prompts/defense.txt").read()
        
        formatted_prompt = prompt_template.format(
            defender_role=self.role_name,
            challenger_role=challenger_role,
            query=query,
            context=context,
            captions=caption,
            text_agent_answer=text_ans,
            visual_agent_answer=visual_ans,
            disputed_image=disputed_image
        )
        # 组装内容 (如果当前是 Visual Agent，可以在这里传入争议图片的 img_paths)
        content = self._build_content(formatted_prompt, img_paths)
        return self._call_llm(content, temperature=0.3)
    
    def critique(self, query, context, caption, text_ans, visual_ans, disputed_image, defender_role, defender_argument, img_paths=None):
        """质询方法"""
        prompt_template = open("prompts/critique.txt").read()
        
        formatted_prompt = prompt_template.format(
            challenger_role=self.role_name,
            defender_role=defender_role,
            query=query,
            context=context,
            captions=caption,
            text_agent_answer=text_ans,
            visual_agent_answer=visual_ans,
            disputed_image=disputed_image,
            defender_argument=defender_argument
        )
        # 同理，支持在质询时查看图片
        content = self._build_content(formatted_prompt, img_paths)
        return self._call_llm(content, temperature=0.3)