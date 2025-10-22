import os, json, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
 
def sanitize_filename(name):
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name)
    if len(name) > 100:
        name = name[:100].rstrip()
    return name

def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # 先頭行の```... を落として末尾の ``` を落とす
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        t = "\n".join(lines).strip()
    return t

def _generate_one_topic_detail(
    client, gen_model, config_text,
    instr_topic_details_generation: str,
    by_sid: dict, topic: dict,
    evidence_dir: Path, draft_dir: Path
):
    start_time_one_topic_detail_generation = time.time()
    idx   = topic.get("idx")
    title = topic.get("title")
    sids  = topic.get("sids", [])

    evidence_rows = []
    missing_here = []
    for sid in sids:
        row = by_sid.get(sid)
        if row is None:
            missing_here.append(sid)
        else:
            evidence_rows.append({
                "sid": row["sid"],
                "text": row["text"],
                "start": row["start"],
                "end": row["end"],
                "role": row["role"],
            })

    if missing_here:
        print(f"  [WARN] {len(missing_here)} SID(s) not found. e.g., {missing_here[:5]}")

    topic_name = sanitize_filename(title or f"topic_{idx}")
    base = f"{idx:02d} - {topic_name}" if isinstance(idx, int) else topic_name

    out_json_path = evidence_dir / f"{base} - evidences.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "idx": idx,
            "title": title,
            "count": len(evidence_rows),
            "sids": [r["sid"] for r in evidence_rows],
            "evidences": evidence_rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"Waiting for response from Gemini API for topic {idx}...")
    contents = [
        instr_topic_details_generation,
        f"Topic: {title}",
        json.dumps(evidence_rows, ensure_ascii=False, indent=2),
        "Using the text and JSON data provided above, follow the instructions and return the result in plain text.",
    ]
    response_topic_detail = client.models.generate_content(
        model=gen_model,
        contents=contents,
        config=config_text
    )

    print("saving response...")
    draft_path = draft_dir / f"{base} - details.txt"
    draft_path.write_text(response_topic_detail.text, encoding="utf-8")

    elapsed = time.time() - start_time_one_topic_detail_generation
    print(f"  --> ⏰ Generated details for topic {idx} in {elapsed:.2f} seconds.")

    return {
        "idx": idx,
        "title": title,
        "json_path": str(out_json_path),
        "count": len(evidence_rows),
        "missing_sids": missing_here,
    }

def _check_one_faithfulness(
    client, gen_model, config_text,
    instr_faithfulness_check: str,
    evidence_path: Path, draft_path: Path, edited_dir: Path
):
    start = time.time()

    # 名前の整合を確認
    if evidence_path.stem.split(" - ")[0] != draft_path.stem.split(" - ")[0]:
        raise ValueError(f"Name mismatch: {evidence_path} vs {draft_path}")

    evidence_data = json.loads(evidence_path.read_text(encoding="utf-8"))
    detail_text   = draft_path.read_text(encoding="utf-8")
    json_data_as_text_evidence = json.dumps(evidence_data, ensure_ascii=False, indent=2)

    print(f"Waiting for response from Gemini API... [{draft_path.name}]")
    contents = [
        instr_faithfulness_check,
        detail_text,
        json_data_as_text_evidence,
        "Using the text(draft markdown) and JSON(evidence) data provided above, follow the instructions and return the result in plain text.",
    ]
    resp = client.models.generate_content(
        model=gen_model,
        contents=contents,
        config=config_text
    )

    out_path = edited_dir / draft_path.name
    out_path.write_text(resp.text, encoding="utf-8")

    elapsed = time.time() - start
    print(f"  --> ⏰ Checked and edited details for {draft_path.name} in {elapsed:.2f} seconds.")
    return out_path


def generate_topic_details(client, gen_model, config_json, config_text, lecture_dir: Path):
    # topicsごとのsidを生成
    print("\n### Topic Evidence Selection ###")

    start_time_topic_evidence_selection = time.time()
    instr_topic_evidence_selection = Path(PROMPTS_DIR / "topic_evidence_selection.txt").read_text(encoding="utf-8")

    with open(lecture_dir / "topics.txt", "r", encoding="utf-8") as f:
        topics = f.read()

    sentences_final = Path(lecture_dir / "sentences_final.json").read_text(encoding="utf-8")

    print("Waiting for response from Gemini API...")
    contents = [
        instr_topic_evidence_selection,
        topics,
        sentences_final,
        "Using the text and JSON data provided above, follow the instructions and return the result in JSON format."
    ]
    response_topic_evidence_selection = client.models.generate_content(
        model = gen_model,
        contents = contents,
        config = config_json
    )

    print("saving response...")
    raw = response_topic_evidence_selection.text
    clean = _strip_code_fence(raw)
    out_topic_evidence_selection = json.loads(clean)

    with open(lecture_dir / "topic_evidence_selections.json", "w", encoding="utf-8") as f:
        json.dump(out_topic_evidence_selection, f, ensure_ascii=False, indent=2)

    end_time_topic_evidence_selection = time.time()
    elapsed_time_topic_evidence_selection = end_time_topic_evidence_selection - start_time_topic_evidence_selection
    print(f"⏰Selected topic evidence: {elapsed_time_topic_evidence_selection:.2f} seconds.")


    # トピックごとに詳細を生成
    print("\n### Topic Details Generation ###")

    start_time_topic_details_generation = time.time()

    max_workers = 5
    index_manifest = []
    total_missing = 0

    EVIDENCE_DIR = Path(lecture_dir / "evidences")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    DETAIL_DIR = Path(lecture_dir / "details")
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    DETAIL_DRAFT_DIR = Path(lecture_dir / "details/drafts")
    DETAIL_DRAFT_DIR.mkdir(parents=True, exist_ok=True)

    DETAIL_EDITED_DIR = Path(lecture_dir / "details/edited")
    DETAIL_EDITED_DIR.mkdir(parents=True, exist_ok=True)

    instr_topic_details_generation = Path(PROMPTS_DIR / "topic_details_generation.txt").read_text(encoding="utf-8")

    with open(lecture_dir / "topic_evidence_selections.json", "r", encoding="utf-8") as f:
        topic_evidence_selections = json.load(f)
    topic_evidence = topic_evidence_selections.get("topics", [])

    with open(lecture_dir / "sentences_final.json", "r", encoding="utf-8") as f:
        sentences_final = json.load(f)

    by_sid = {row["sid"]: row for row in sentences_final}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        submit_one = partial(
            _generate_one_topic_detail,
            client, gen_model, config_text,
            instr_topic_details_generation,
            by_sid,
            evidence_dir=EVIDENCE_DIR,
            draft_dir=DETAIL_DRAFT_DIR
        )
        futures = {ex.submit(submit_one, topic): topic for topic in topic_evidence}

        for fut in as_completed(futures):
            topic = futures[fut]
            try:
                manifest_entry = fut.result()
                index_manifest.append(manifest_entry)
                total_missing += len(manifest_entry.get("missing_sids", []))
            except Exception as e:
                print(f"❌ Topic idx={topic.get('idx')} failed: {e}")

    end_time_topic_details_generation = time.time()
    elapsed_time_topic_details_generation = end_time_topic_details_generation - start_time_topic_details_generation
    print(f"⏰Generated topic details: {elapsed_time_topic_details_generation:.2f} seconds.")


    # 生成された詳細の忠実性チェックと最小限の修正
    print("\n### Faithfulness Check and Minimal Edit###")
    start_time_faithfulness_check = time.time()

    instr_faithfulness_check = Path(PROMPTS_DIR / "faithfulness_check_and_minimal_edit.txt").read_text(encoding="utf-8")

    evidence_files = sorted(EVIDENCE_DIR.glob("* - evidences.json"))
    if not evidence_files:
        raise RuntimeError("no JSON in evidences/")
    detail_files = sorted(DETAIL_DRAFT_DIR.glob("* - details.txt"))
    if not detail_files:
        raise RuntimeError("no text file in details/")

    if len(evidence_files) != len(detail_files):
        raise RuntimeError(f"Count mismatch: {len(evidence_files)} vs {len(detail_files)}")

    def _prefix(p: Path) -> str:
        return p.stem.split(" - ")[0]
    
    ev_by_prefix = { _prefix(p): p for p in evidence_files }
    dt_by_prefix = { _prefix(p): p for p in detail_files }
    common_keys = sorted(set(ev_by_prefix) & set(dt_by_prefix))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        submit_one = partial(
            _check_one_faithfulness,
            client, gen_model, config_text,
            instr_faithfulness_check,
            edited_dir=DETAIL_EDITED_DIR
        )
        futures = {
            ex.submit(submit_one, ev_by_prefix[k], dt_by_prefix[k]): k
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
    LECTURE_DIR = ROOT / "lectures/2025-10-07-21-27-43-0700"

    generate_topic_details(client, GEN_MODEL, config_json(), config_text(), LECTURE_DIR)

if __name__ == "__main__":
    main()