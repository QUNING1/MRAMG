from openai import OpenAI
import json
import yaml
import base64
from pathlib import Path
class BaseAgent:
    def __init__(self, client: OpenAI, model: str, role_name: str, temperature: float = 0.3):
        if client is None:
            raise ValueError("OpenAI client must be provided")

        self.client = client
        self.model = model
        self.role_name = role_name
        self.temperature = temperature

        # 在初始化时一次性加载 YAML 配置
        with open("prompts/conflict_templates.yaml", "r", encoding="utf-8") as f:
            self.templates_config = yaml.safe_load(f)

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
    def defend(self, query, context, caption, text_ans, visual_ans, disputed_item, challenger_role, conflict_type, img_paths=None):
        """举证方法"""
        prompt_template = open("prompts/defense.txt").read()
        defender_name = self.role_name.lower().replace(" ", "_")
        challenger_name = challenger_role.lower().replace(" ", "_")

        # CONFLICT_TEMPLATES = {
        #     "set_conflict": {
        #         "conflict_description": """We are currently resolving a conflict regarding the image placeholder: {disputed_image}. \nAs the {defender_role}, you chose to use {disputed_image}, while the {challenger_role} did not. """,
        #         # "task_description": """Your task is to DEFEND your decision to use {disputed_image}. 
        #         # Based STRICTLY on the provided Query, Context, and Image Captions, explain why inserting this specific image adds necessary value to the final output."""
        #     },
        #     "order_conflict": {
        #         "conflict_description": """We are currently resolving a sequential conflict regarding the ordering of images.\nBoth agents agreed to include these images, but disagreed on their chronological or logical sequence: \n- Your proposed sequence ({defender_role}): {defender_img_order} \n- The {challenger_role}'s proposed sequence: {challenger_img_order}""",
        #     }
        # }
        try:
            conflict_desc_template = self.templates_config["defense"][conflict_type]["conflict_description"]
        except KeyError:
            raise ValueError(f"Template not found for action 'defense' and conflict_type '{conflict_type}'")

        # 验证disputed_item必须为dict
        if not isinstance(disputed_item, dict):
            raise ValueError("disputed_item must be a dictionary")
        
        if conflict_type == "set_conflict" and disputed_item.get("disputed_image") is None:
            raise ValueError("disputed_item must contain 'disputed_image' key")

        if conflict_type == "order_conflict" and (disputed_item.get(f"{defender_name}_img_order") is None or disputed_item.get(f"{challenger_name}_img_order") is None):
            raise ValueError(f"disputed_item must contain '{defender_name}_img_order' and '{challenger_name}_img_order' keys")
        
        conflict_desc = conflict_desc_template.format(
            disputed_image=disputed_item.get("disputed_image"),
            defender_role=self.role_name,
            challenger_role=challenger_role, 
            defender_img_order=disputed_item.get(f"{defender_name}_img_order"),
            challenger_img_order=disputed_item.get(f"{challenger_name}_img_order"),
        )

        formatted_prompt = prompt_template.format(
            defender_role=self.role_name,
            query=query,
            context=context,
            caption=caption,
            text_agent_answer=text_ans,
            visual_agent_answer=visual_ans,
            disputed_image=disputed_item.get("disputed_image"),
            conflict_description=conflict_desc,
        )
        # 组装内容 (如果当前是 Visual Agent，可以在这里传入争议图片的 img_paths)
        content = self._build_content(formatted_prompt, img_paths)
        return self._call_llm(content, temperature=0.3)
    
    def critique(self, query, context, caption, text_ans, visual_ans, disputed_item, defender_role, defender_argument, conflict_type, img_paths=None):
        """质询方法"""
        prompt_template = open("prompts/critique.txt").read()
        challenger_name = self.role_name.lower().replace(" ", "_")
        defender_name = defender_role.lower().replace(" ", "_")
        
        # 2. 根据冲突类型，获取对应的子文案
        try:
            conflict_desc_template = self.templates_config["critique"][conflict_type]["conflict_description"]
        except KeyError:
            raise ValueError(f"Template not found for action 'critique' and conflict_type '{conflict_type}'")

         # 验证disputed_item必须为dict
        if not isinstance(disputed_item, dict):
            raise ValueError("disputed_item must be a dictionary")
        
        if conflict_type == "set_conflict" and disputed_item.get("disputed_image") is None:
            raise ValueError("disputed_item must contain 'disputed_image' key")

        if conflict_type == "order_conflict" and (disputed_item.get(f"{defender_name}_img_order") is None or disputed_item.get(f"{challenger_name}_img_order") is None):
            raise ValueError(f"disputed_item must contain '{defender_name}_img_order' and '{challenger_name}_img_order' keys")

        conflict_desc = conflict_desc_template.format(
            disputed_image=disputed_item.get("disputed_image"),
            defender_role=defender_role,
            challenger_role=self.role_name, 
            defender_img_order=disputed_item.get(f"{defender_name}_img_order"),
            challenger_img_order=disputed_item.get(f"{challenger_name}_img_order"),
        )

        formatted_prompt = prompt_template.format(
            challenger_role=self.role_name,
            defender_role=defender_role,
            query=query,
            context=context,
            caption=caption,
            text_agent_answer=text_ans,
            visual_agent_answer=visual_ans,
            defender_argument=defender_argument, 
            conflict_description=conflict_desc,
        )
        # 同理，支持在质询时查看图片
        content = self._build_content(formatted_prompt, img_paths)
        return self._call_llm(content, temperature=0.3)