import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

with open("prompts/gemini_prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

with open("transcription/AI_and_ethics_10_copy_with_gpt4o.txt", "r", encoding="utf-8") as f:
    transcription_text = f.read()

full_prompt = f"{prompt}\n\nTranscription:\n{transcription_text}"

response = model.generate_content(full_prompt)

output_path = "summary_AI_and_ethics_10_copy.txt"

# 出力をファイルに保存
with open(output_path, "w", encoding="utf-8") as f:
    f.write(response.text)