import json, re, time, math, asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

SID_NUM = re.compile(r"s(\d+)")


def topic_extraction_draft(client, gen_model, config_text, lecture_dir: Path):
    # トピックを選出
    print("\n### Topic Extraction ###")
    start_time_topic_extraction = time.time()

    with open(PROMPTS_DIR / "topic_extraction.txt", "r", encoding="utf-8") as f:
        instr_topic_extraction = f.read()

    with open(lecture_dir / "sentences_final.json", "r", encoding="utf-8") as f:
        sentences_final = json.load(f)

    print("Waiting for response from Gemini API...")

    payload = {
        "task": "Topic Extraction",
        "instruction": instr_topic_extraction,
        "data": {
            "sentences": sentences_final
        }
    }

    contents = [
        "This is very important task, but I am sure that you will do this well, because you are the best data processer. Read the JSON and follow the instructions carefully.",
        json.dumps(payload, ensure_ascii=False)
    ]

    response_extract_topic = client.models.generate_content(
        model = gen_model,
        contents = contents,
        config = config_text,
    )

    print("saving response...")
    with open(lecture_dir / "topics.txt", "w", encoding="utf-8") as f:
        f.write(response_extract_topic.text)

    end_time_topic_extraction = time.time()
    elapsed_time_topic_extraction = end_time_topic_extraction - start_time_topic_extraction
    print(f"⏰Extracted topic: {elapsed_time_topic_extraction:.2f} seconds.")


def topic_extraction(client, gen_model, gen_model_lite, config_json, lecture_dir: Path):

    topic_extraction_draft(client, gen_model, config_json, lecture_dir)

    print("\n✅All tasks of TOPIC EXTRACTION completed.")


# ------ for test -------
def config_json(thinking: int = 0, google_search: bool = False):
    kwargs = dict(
        temperature=0.2,
        response_mime_type="application/json",
    )
    if thinking > 0:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking)
    if google_search:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    return types.GenerateContentConfig(**kwargs)

def config_text(thinking: int = 0, google_search: int = 0):
    kwargs = dict(
        temperature=0.2,
        response_mime_type="text/plain",
    )
    if thinking > 0:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking)
    if google_search:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    return types.GenerateContentConfig(**kwargs)

def main():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    flash = "gemini-2.5-flash"
    flash_lite = "gemini-2.5-flash-lite"

    ROOT = Path(__file__).resolve().parent
    LECTURE_DIR = ROOT / "../lectures/2025-10-31-12-04-37-0700"  # ⚠️ CHANGE FOLDER NAME!!! 🛑

    topic_extraction(client, flash, flash_lite, config_json(), config_text(), LECTURE_DIR)

    # sentence_review(client, flash, config_json(), LECTURE_DIR)
    
    # role_classification(client, flash_lite, config_json(), LECTURE_DIR, 300, 10)

    # role_review(client, flash, config_json(), LECTURE_DIR)

    # topic_extraction(client, flash, config_text(), LECTURE_DIR)

if __name__ == "__main__":
    main()