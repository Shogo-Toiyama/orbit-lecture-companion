import os, json, time, re
from pathlib import Path
 
def sanitize_filename(name):
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name)
    if len(name) > 100:
        name = name[:100].rstrip()
    return name

def generate_topic_details(model_json, model_text, lecture_dir: Path):
    # topicsごとのsidを生成
    print("\n### Topic Evidence Selection ###")

    start_time_topic_evidence_selection = time.time()
    with open("prompts/topic_evidence_selection.txt", "r", encoding="utf-8") as f:
        instr_topic_evidence_selection = f.read()

    with open(lecture_dir / "topics.txt", "r", encoding="utf-8") as f:
        topics = f.read()

    with open(lecture_dir / "sentences_final.json", "r", encoding="utf-8") as f:
        sentences_final = json.load(f)

    json_data_as_text_sentences_final = json.dumps(sentences_final, ensure_ascii=False, indent=2)

    print("Waiting for response from Gemini API...")
    response_topic_evidence_selection = model_json.generate_content([
        instr_topic_evidence_selection,
        topics,
        json_data_as_text_sentences_final,
        "Using the text and JSON data provided above, follow the instructions and return the result in JSON format."
    ])

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
        response_topic_detail = model_text.generate_content([
            instr_topic_details_generation,
            f"Topic: {title}",
            json.dumps(evidence_rows, ensure_ascii=False, indent=2),
            "Using the text and JSON data provided above, follow the instructions and return the result in plain text.",
        ])

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
        response_faithfulness_check = model_text.generate_content([
            instr_faithfulness_check,
            detail_text,
            json_data_as_text_evidence,
            "Using the text(draft markdown) and JSON(evidence) data provided above, follow the instructions and return the result in plain text.",
        ])

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