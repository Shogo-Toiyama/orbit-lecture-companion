import json, re, time, math, asyncio
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

SID_NUM = re.compile(r"s(\d+)")

def token_report(resp):
    um = resp.usage_metadata
    prompt_tokens = um.prompt_token_count
    candidate_tokens = um.candidates_token_count
    total_tokens = um.total_token_count
    thinking_tokens = total_tokens - prompt_tokens - candidate_tokens
    return (f"TOKEN USAGE REPORT\n  ⬆️:{prompt_tokens}, 🧠: {thinking_tokens}, ⬇️: {candidate_tokens}\n  TOTAL: {total_tokens}")

def sid_to_num(sid: str):
    m = SID_NUM.match(sid)
    if m:
        return int(m.group(1))
    else:
        return None

def topic_segmentation(client, gen_model, config_json, lecture_dir: Path):
    # トピックで分割
    print("\n### Topic Segmentation ###")
    start_time_topic_segmentation = time.time()

    with open(PROMPTS_DIR / "topic_segmentation.txt", "r", encoding="utf-8") as f:
        instr_topic_segmentation = f.read()

    with open(lecture_dir / "sentences_final.json", "r", encoding="utf-8") as f:
        sentences_final = json.load(f)

    print("Waiting for response from Gemini API...")

    payload = {
        "task": "Topic Segmentation",
        "instruction": instr_topic_segmentation,
        "data": {
            "sentences": sentences_final
        }
    }

    contents = [
        "This is very important task, but I am sure that you will do this well, because you are the best data processer. Read the JSON and follow the instructions carefully.",
        json.dumps(payload, ensure_ascii=False)
    ]

    response_topic_segmentation = client.models.generate_content(
        model = gen_model,
        contents = contents,
        config = config_json,
    )

    print("saving response...")
    with open(lecture_dir / "topic_segments.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(response_topic_segmentation.text.strip()), f, ensure_ascii=False, indent=2)

    end_time_topic_segmentation = time.time()
    elapsed_time_topic_segmentation = end_time_topic_segmentation - start_time_topic_segmentation
    print(token_report(response_topic_segmentation))
    print(f"⏰Extracted topic: {elapsed_time_topic_segmentation:.2f} seconds.")

def out_of_segment_classification(client, gen_model_lite, config_json, lecture_dir: Path):
    # セグメント外の文章分類
    print("\n### Out of Segments Classification ###")
    start_time_oos_classification = time.time()

    with open(lecture_dir / "topic_segments.json", "r", encoding="utf-8") as f:
        topic_segments = json.load(f)

    with open(lecture_dir / "sentences_final.json", "r", encoding="utf-8") as f:
        sentences_final = json.load(f)

    instr_oos_classification = Path(PROMPTS_DIR / "out_of_segment_classification.txt").read_text(encoding="utf-8")

    segment_sids = [(sid_to_num(t["start_sid"]), sid_to_num(t["end_sid"])) for t in topic_segments["topics", []]]
    last_sid = len(sentences_final)
    out_of_segment_sids = []
    current_sid = 1
    for start_sid, end_sid in segment_sids:
        if (current_sid == start_sid):
            current_sid = end_sid + 1
        elif (current_sid < start_sid):
            out_of_segment_sids.append((current_sid, start_sid-1))
            current_sid = end_sid + 1
        else:
            raise ValueError("Segments are overlapping or not sorted correctly.")
    if current_sid <= last_sid:
        out_of_segment_sids.append((current_sid, last_sid))

    payload = {
        "task": "Out of Segment Classification",
        "instruction": instr_oos_classification,
        "data": {
            "sentences": sentences_final,
            "out_of_segment_sids": out_of_segment_sids
        }
    }

    contents = [
        "This is very important task, but I am sure that you will do this well, because you are the best data processer. Read the JSON and follow the instructions carefully.",
        json.dumps(payload, ensure_ascii=False)
    ]

    response_oos_classification = client.models.generate_content(
        model = gen_model_lite,
        contents = contents,
        config = config_json,
    )

    print("saving response...")
    with open(lecture_dir / "out_of_segment_classification.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(response_oos_classification.text.strip()), f, ensure_ascii=False, indent=2)

    end_time_oos_classification = time.time()
    elapsed_time_oos_classification = end_time_oos_classification - start_time_oos_classification
    print(token_report(response_oos_classification))
    print(f"⏰Classified out of segments: {elapsed_time_oos_classification:.2f} seconds.")
    

def lecture_segmentation(client, gen_model, gen_model_lite, config_json, lecture_dir: Path):

    topic_segmentation(client, gen_model, config_json, lecture_dir)

    print("\n✅All tasks of LECTURE SEGMENTATION completed.")


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

    lecture_segmentation(client, flash, flash_lite, config_json(), LECTURE_DIR)

    # sentence_review(client, flash, config_json(), LECTURE_DIR)
    
    # role_classification(client, flash_lite, config_json(), LECTURE_DIR, 300, 10)

    # role_review(client, flash, config_json(), LECTURE_DIR)

    # topic_extraction(client, flash, config_text(), LECTURE_DIR)

if __name__ == "__main__":
    main()