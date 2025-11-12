import os, json, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
_ILLEGAL_FS = re.compile(r'[\\/:*?"<>|\n\r\t]')

def token_report(resp):
    um = resp.usage_metadata
    prompt_tokens = um.prompt_token_count
    candidate_tokens = um.candidates_token_count
    total_tokens = um.total_token_count
    thinking_tokens = total_tokens - prompt_tokens - candidate_tokens
    return (f"TOKEN USAGE REPORT\n  ⬆️:{prompt_tokens}, 🧠: {thinking_tokens}, ⬇️: {candidate_tokens}\n  TOTAL: {total_tokens}")

def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        t = "\n".join(lines).strip()
    return t

def _safe_filename(name: str, max_len: int = 120) -> str:
    name = _ILLEGAL_FS.sub('_', name).strip()
    return (name[:max_len]).rstrip(' .')

def _slice_by_sid(sentences, start_sid=None, end_sid=None):
    if isinstance(sentences, dict):
        seq = sorted(sentences.values(), key=lambda s: s.get("sid", ""))
    else:
        seq = list(sentences)
    if not seq:
        return []

    sid_to_idx = {}
    for i, s in enumerate(seq):
        sid = s.get("sid")
        if sid is not None and sid not in sid_to_idx:
            sid_to_idx[sid] = i

    if start_sid is None:
        i0 = 0
    else:
        if start_sid not in sid_to_idx:
            raise ValueError(f"start_sid not found: {start_sid}")
        i0 = sid_to_idx[start_sid]

    if end_sid is None:
        i1 = len(seq) - 1
    else:
        if end_sid not in sid_to_idx:
            raise ValueError(f"end_sid not found: {end_sid}")
        i1 = sid_to_idx[end_sid]

    if i0 > i1:
        i0, i1 = i1, i0

    if i0 >= 20:
        i0 -= 20
    else:
        i0 = 0
    if i1 + 20 < len(seq):
        i1 += 20
    else:
        i1 = len(seq) - 1

    return seq[i0:i1+1]

def _generate_one_topic_detail(
    client, gen_model, config_text,
    instr_topic_details_generation: str,
    details_dir: Path, sentences, segment:dict
):
    start_time_one_topic_detail_generation = time.time()
    idx = segment["idx"]
    title = segment.get("title", f"Topic {idx}")
    start_sid = segment.get("start_sid")
    end_sid = segment.get("end_sid")

    ALLOWED = ["sid", "text", "role"]
    projected_sentences = [{k: s.get(k) for k in ALLOWED} for s in sentences]

    partial_sentences = _slice_by_sid(projected_sentences, start_sid, end_sid)

    print(f"Waiting for response from Gemini API for topic {idx}...")

    payload = {
        "task": "Topic Detail Generation",
        "instruction": instr_topic_details_generation,
        "data": {
            "topic": segment,
            "partial-transcript": partial_sentences
        }
    }

    contents = [
        "This is very important task, but I am sure that you will do this well, because you are the best data processer. Read the JSON and follow the instructions carefully.",
        json.dumps(payload, ensure_ascii=False)
    ]

    response_topic_detail = client.models.generate_content(
        model=gen_model,
        contents=contents,
        config=config_text
    )

    print("saving response...")
    details_path = details_dir / f"{idx} - {_safe_filename(title)} - details.txt"
    details_path.write_text(response_topic_detail.text.strip(), encoding="utf-8")

    elapsed = time.time() - start_time_one_topic_detail_generation
    print(token_report(response_topic_detail))
    print(f"  --> ⏰ Generated details for topic {idx} in {elapsed:.2f} seconds.")

def _check_one_faithfulness(
    client, gen_model, config_text,
    instr_faithfulness_check: str, sentences: dict, edited_dir: Path,
    segment: dict, draft_path: Path
):
    start = time.time()

    # 名前の整合を確認
    if str(segment.get("idx")).zfill(2) != draft_path.stem.split(" - ")[0].zfill(2):
        raise ValueError(f"Name mismatch: {segment.get("idx")} vs {draft_path}")

    detail_text = draft_path.read_text(encoding="utf-8")

    print(f"Waiting for response from Gemini API... [{draft_path.name}]")

    payload = {
        "task": "Faithfulness Check and Readability Enhancement",
        "instruction": instr_faithfulness_check,
        "data": {
            "detail-draft": detail_text,
            "topic-segment": segment,
            "full-transcript": sentences
        }
    }

    contents = [
        "This is very important task, but I am sure that you will do this well, because you are the best data processer. Read the JSON and follow the instructions carefully.",
        json.dumps(payload, ensure_ascii=False)
    ]
    resp = client.models.generate_content(
        model=gen_model,
        contents=contents,
        config=config_text
    )

    out_path = edited_dir / draft_path.name
    out_path.write_text(resp.text, encoding="utf-8")

    elapsed = time.time() - start
    print(token_report(resp))
    print(f"  --> ⏰ Checked and edited details for {draft_path.name} in {elapsed:.2f} seconds.")
    return out_path

def generate_details_draft(client, gen_model, config_json, lecture_dir: Path):
# トピックごとに詳細を生成
    print("\n### Topic Details Generation ###")

    start_time_topic_details_generation = time.time()

    max_workers = 3

    DETAILS_DIR = Path(lecture_dir / "details")
    DETAILS_DIR.mkdir(exist_ok=True, parents=True)

    instr_topic_details_generation = Path(PROMPTS_DIR / "topic_details_generation_from_segments.txt").read_text(encoding="utf-8")

    with open(lecture_dir / "topic_segments.json", "r", encoding="utf-8") as f:
        topic_segments_json = json.load(f)
    topic_segments = topic_segments_json.get("topics", [])

    with open(lecture_dir / "sentences_final.json", "r", encoding="utf-8") as f:
        sentences_final = json.load(f)

    submit_one = partial(
            _generate_one_topic_detail,
            client, gen_model, config_json,
            instr_topic_details_generation,
            DETAILS_DIR, sentences_final
        )
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(submit_one, topic): topic for topic in topic_segments}
        for fut in as_completed(futures):
            seg = futures[fut]
            idx = seg.get("idx")
            try:
                fut.result()
            except Exception as e:
                print(f"[{idx}] ❌ Unhandled error: {e}")

    end_time_topic_details_generation = time.time()
    elapsed_time_topic_details_generation = end_time_topic_details_generation - start_time_topic_details_generation
    print(f"⏰Generated topic details: {elapsed_time_topic_details_generation:.2f} seconds.")

def faithfulness_check_and_readablity_enhancement(client, gen_model, config_text, lecture_dir: Path):
    # 生成された詳細の忠実性チェックと最小限の修正
    print("\n### Faithfulness Check and Minimal Edit###")
    start_time_faithfulness_check = time.time()
    max_workers = 5
    DETAIL_DRAFT_DIR = Path(lecture_dir / "details/drafts")
    DETAIL_EDITED_DIR = Path(lecture_dir / "details/edited")

    instr_faithfulness_check = Path(PROMPTS_DIR / "faithfulness_check_and_minimal_edit.txt").read_text(encoding="utf-8")

    with open(lecture_dir / "sentences_final.json", "r", encoding="utf-8") as f:
        sentences_final = json.load(f)

    with open(lecture_dir / "topic_segments.json", "r", encoding="utf-8") as f:
        topic_segments_json = json.load(f)
    segments = [ t for t in topic_segments_json.get("topics", []) ]
    detail_files = sorted(DETAIL_DRAFT_DIR.glob("* - details.txt"))
    if not detail_files:
        raise RuntimeError("no text file in details/")

    if len(segments) != len(detail_files):
        raise RuntimeError(f"Count mismatch: {len(segments)} vs {len(detail_files)}")

    def _prefix(p: Path) -> str:
        return p.stem.split(" - ")[0]
    
    seg_by_idx = { str(seg["idx"]).zfill(2): seg for seg in segments }
    print(f"Seg by Idx {seg_by_idx}")
    dt_by_prefix = { _prefix(p).zfill(2): p for p in detail_files }
    print(f"DT by Prefix {dt_by_prefix}")
    common_keys = sorted(set(seg_by_idx) & set(dt_by_prefix))
    print(f"Found {len(common_keys)}: {common_keys}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        submit_one = partial(
            _check_one_faithfulness,
            client, gen_model, config_text,
            instr_faithfulness_check, sentences_final, DETAIL_EDITED_DIR
        )
        futures = {
            ex.submit(submit_one, seg_by_idx[k], dt_by_prefix[k]): k
            for k in common_keys
        }

        for fut in as_completed(futures):
            k = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"❌ Faithfulness failed for {k}: {e}")
                
    end_time_faithfulness_check = time.time()
    elapsed_time_faithfulness_check = end_time_faithfulness_check - start_time_faithfulness_check
    print(f"⏰Checked and edited topic details: {elapsed_time_faithfulness_check:.2f} seconds.")


def generate_topic_details(client, gen_model, config_json, config_text, lecture_dir: Path):
    
    generate_details_draft(client, gen_model, config_text, lecture_dir)
    
    # faithfulness_check_and_readablity_enhancement(client, gen_model, config_text, lecture_dir)

    print("\n✅All tasks of TOPIC DETAIL GENERATION completed.")


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

    GEN_MODEL = "gemini-2.5-flash"

    ROOT = Path(__file__).resolve().parent
    LECTURE_DIR = ROOT / "../lectures/2025-11-11-16-38-54-0800"

    generate_topic_details(client, GEN_MODEL, config_json(), config_text(), LECTURE_DIR)

if __name__ == "__main__":
    main()