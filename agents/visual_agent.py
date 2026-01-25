def visual_answer(llm, images):
    # 第一版：用 caption / OCR 当 visual reasoning
    prompt = open("prompts/visual.txt").read()
    return llm(prompt.format(images=images))