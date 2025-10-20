import os, json, time, re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
 
def sanitize_filename(name):
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name)
    if len(name) > 100:
        name = name[:100].rstrip()
    return name

def generate_topic_details(client, gen_model, config_json, config_text, lecture_dir: Path):
    # topicsごとのsidを生成
    print("\n### Topic Evidence Selection ###")

    start_time_topic_evidence_selection = time.time()
    instr_topic_evidence_selection = Path("prompts/topic_evidence_selection.txt").read_text(encoding="utf-8")

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
    out_topic_evidence_selection = json.loads(response_topic_evidence_selection.text)

    with open(lecture_dir / "topic_evidence_selections.json", "w", encoding="utf-8") as f:
        json.dump(out_topic_evidence_selection, f, ensure_ascii=False, indent=2)

    end_time_topic_evidence_selection = time.time()
    elapsed_time_topic_evidence_selection = end_time_topic_evidence_selection - start_time_topic_evidence_selection
    print(f"⏰Selected topic evidence: {elapsed_time_topic_evidence_selection:.2f} seconds.")


    # トピックごとに詳細を生成
    print("\n### Topic Details Generation ###")

    start_time_topic_details_generation = time.time()

    EVIDENCE_DIR = Path(lecture_dir / "evidences")
    EVIDENCE_DIR.mkdir()

    DETAIL_DIR = Path(lecture_dir / "details")
    DETAIL_DIR.mkdir()

    DETAIL_DRAFT_DIR = Path(lecture_dir / "details/drafts")
    DETAIL_DRAFT_DIR.mkdir()

    DETAIL_EDITED_DIR = Path(lecture_dir / "details/edited")
    DETAIL_EDITED_DIR.mkdir()

    with open("prompts/topic_details_generation.txt", "r", encoding="utf-8") as f:
        instr_topic_details_generation = f.read()

    with open(lecture_dir / "topic_evidence_selections.json", "r", encoding="utf-8") as f:
        topic_evidence_selections = json.load(f)
    topic_evidence = topic_evidence_selections.get("topics", [])

    with open(lecture_dir / "sentences_final.json", "r", encoding="utf-8") as f:
        sentences_final = json.load(f)

    by_sid = {row["sid"]: row for row in sentences_final}

    index_manifest = []
    total_missing = 0

    for topic in topic_evidence:
        start_time_one_topic_detail_generation = time.time()
        idx = topic.get("idx")
        title = topic.get("title")
        sids = topic.get("sids", [])

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
            total_missing += len(missing_here)

        topic_name = sanitize_filename(title or f"topic_{idx}")
        base = f"{idx:02d} - {topic_name}" if isinstance(idx, int) else topic_name

        out_json_path = EVIDENCE_DIR / f"{base} - evidences.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "idx": idx,
                "title": title,
                "count": len(evidence_rows),
                "sids": [r["sid"] for r in evidence_rows],
                "evidences": evidence_rows,
            }, f, ensure_ascii=False, indent=2)
        
        index_manifest.append({
            "idx": idx,
            "title": title,
            "json_path": str(out_json_path),
            "count": len(evidence_rows),
            "missing_sids": missing_here,
        })

        print(f"Waiting for response from Gemini API for topic {idx}...")
        contents = [
            instr_topic_details_generation,
            f"Topic: {title}",
            json.dumps(evidence_rows, ensure_ascii=False, indent=2),
            "Using the text and JSON data provided above, follow the instructions and return the result in plain text.",
        ]
        response_topic_detail = client.models.generate_content(
            model = gen_model,
            contents = contents,
            config = config_text
        )

        print("saving response...")
        with open(DETAIL_DRAFT_DIR / f"{base} - details.txt", "w", encoding="utf-8") as f:
            f.write(response_topic_detail.text)
        end_time_one_topic_detail_generation = time.time()
        elapsed_time_one_topic_detail_generation = end_time_one_topic_detail_generation - start_time_one_topic_detail_generation
        print(f"  --> ⏰ Generated details for topic {idx} in {elapsed_time_one_topic_detail_generation:.2f} seconds.")

    end_time_topic_details_generation = time.time()
    elapsed_time_topic_details_generation = end_time_topic_details_generation - start_time_topic_details_generation
    print(f"⏰Generated topic details: {elapsed_time_topic_details_generation:.2f} seconds.")


    # 生成された詳細の忠実性チェックと最小限の修正
    print("\n### Faithfulness Check and Minimal Edit###")
    start_time_faithfulness_check = time.time()

    with open("prompts/faithfulness_check_and_minimal_edit.txt", "r", encoding="utf-8") as f:
        instr_faithfulness_check = f.read()

    evidence_files = sorted(EVIDENCE_DIR.glob("* - evidences.json"))
    if not evidence_files:
        print("[ERROR] no JSON in evidences/")
        exit(1)
    detail_files = sorted(DETAIL_DRAFT_DIR.glob("* - details.txt"))
    if not detail_files:
        print("[ERROR] no text file in details/")
        exit(1)

    if len(evidence_files) != len(detail_files):
        print(f"[WARN] JSONとテキストファイルの数が一致しません: {len(evidence_files)} vs {len(detail_files)}")
        exit(1)

    for i in range(len(evidence_files)):
        start_time_one_faithfulness_check = time.time()
        evidence_path = evidence_files[i]
        detail_path = detail_files[i]
        if evidence_path.stem.split(" - ")[0] != detail_path.stem.split(" - ")[0]:
            print(f"[WARN] The JSON and text file do not match: {evidence_path} vs {detail_path}")
            exit(1)
        with open(evidence_path, "r", encoding="utf-8") as f:
            evidence_data = json.load(f)
        with open(detail_path, "r", encoding="utf-8") as f:
            detail_text = f.read()

        json_data_as_text_evidence = json.dumps(evidence_data, ensure_ascii=False, indent=2)

        print("Waiting for response from Gemini API...")
        contents = [
            instr_faithfulness_check,
            detail_text,
            json_data_as_text_evidence,
            "Using the text(draft markdown) and JSON(evidence) data provided above, follow the instructions and return the result in plain text.",
        ]
        response_faithfulness_check = client.models.generate_content(
            model = gen_model,
            contents = contents,
            config = config_text
        )

        print("saving response...")
        with open(DETAIL_EDITED_DIR / detail_path.name, "w", encoding="utf-8") as f:
            f.write(response_faithfulness_check.text)
        end_time_one_faithfulness_check = time.time()
        elapsed_time_one_faithfulness_check = end_time_one_faithfulness_check - start_time_one_faithfulness_check
        print(f"  --> ⏰ Checked and edited details for {detail_path.name} in {elapsed_time_one_faithfulness_check:.2f} seconds.")

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