import json, time, math
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
import os

def response_to_text(resp):
    """google-generativeai のレスポンスから安全に text を取り出す。"""
    try:
        return resp.text
    except Exception:
        # 念のため candidates 経由でも拾う
        try:
            cand = resp.candidates[0] if getattr(resp, "candidates", None) else None
            parts = getattr(getattr(cand, "content", None), "parts", None)
            if parts:
                chunks = []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        chunks.append(t)
                if chunks:
                    return "\n".join(chunks)
        except Exception:
            pass
        raise

def debug_dump_response(resp, path_prefix):
    try:
        # finish_reason を見て未完了検知
        fr = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
    except Exception:
        fr = None
    raw = getattr(resp, "text", None) or ""
    with open(f"{path_prefix}_raw.txt", "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"[DEBUG] finish_reason={fr}, saved raw -> {path_prefix}_raw.txt")

def split_balanced(n_items: int, max_batch: int):
    if n_items <= 0:
        return []
    n_batches = math.ceil(n_items / max_batch)
    base = n_items // n_batches
    rem = n_items % n_batches  # 先頭 rem バッチに +1
    ranges = []
    start = 0
    for i in range(n_batches):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges

def make_context_slice(n_items: int, start: int, end: int, ctx: int):
    """ターゲット [start,end) の前後に ctx 文の文脈を付けた [cstart,cend) を返す"""
    return max(0, start - ctx), min(n_items, end + ctx)

def call_model_json_with_retry(model, prompt_texts, out_prefix, tries=3, sleep_sec=2.0):
    """
    model.generate_content(prompt_texts) をリトライ付きで叩き、
    生テキストをデバッグ保存して JSON を返す。
    """
    last_err = None
    for t in range(1, tries+1):
        try:
            resp = model.generate_content(prompt_texts)
            debug_dump_response(resp, out_prefix)
            txt = response_to_text(resp)
            # 念のため末尾の未完やトレーリングを検出しつつそのまま parse
            out = json.loads(txt)
            return out
        except Exception as e:
            last_err = e
            print(f"[WARN] model call failed (try {t}/{tries}): {e}")
            time.sleep(sleep_sec)
    raise RuntimeError(f"All attempts failed. Last error: {last_err}")


def topic_extraction(model_json, model_text, lecture_dir: Path, max_batch_size: int = 450, ctx: int = 10):
    # sentencesからrole分類
    print("\n### Role Classification ###")
    start_time_role_classification = time.time()

    with open("prompts/role_classification.txt", "r", encoding="utf-8") as f:
        instr_role_classification = f.read()

    with open(lecture_dir / "transcript_sentences.json", "r", encoding="utf-8") as f:
        sentences = json.load(f)

    ALLOWED = {"sid", "text", "start", "end", "speaker"}

    projected = [{k: s.get(k) for k in ALLOWED} for s in sentences]
    n = len(projected)
    print(f"[INFO] sentences: {n}")

    ranges = split_balanced(n, max_batch_size)
    print(f"[INFO] batches: {len(ranges)} -> {ranges}")

    batch_dir = lecture_dir / "role_batches"
    batch_dir.mkdir(exist_ok=True)

    all_labels = []
    missing_total = []
    for bi, (start, end) in enumerate(ranges, start=1):
        cstart, cend = make_context_slice(n, start, end, ctx=ctx)
        payload = projected[cstart:cend]

        start_sid = projected[start]["sid"]
        end_sid = projected[end-1]["sid"]
        target_len = end - start

        prefix = batch_dir / f"batch_{bi:02d}"

        # 入力保存（デバッグ）
        with open(f"{prefix}_payload.json", "w", encoding="utf-8") as f:
            json.dump({
                "range": {"start": start, "end": end, "cstart": cstart, "cend": cend,
                          "start_sid": start_sid, "end_sid": end_sid},
                "payload": payload
            }, f, ensure_ascii=False, indent=2)

        print(f"[BATCH {bi}] send: target={target_len} ctx=({cstart},{cend}) sids=[{start_sid}..{end_sid}]")

        # 呼び出し（テキスト渡し）
        prompt_texts = [
            instr_role_classification,
            json.dumps(payload, ensure_ascii=False, indent=2),
            "Using the JSON above, label ONLY the target range and return JSON."
        ]

        out = call_model_json_with_retry(
            model_json,
            prompt_texts,
            out_prefix=str(prefix)
        )

        # 出力の形をチェック
        labels = out.get("labels", [])
        # ターゲット sids を列挙してマップ
        target_sids = [projected[i]["sid"] for i in range(start, end)]

        # フィルタ：ターゲット範囲以外が混ざってたら落とす
        filtered = [lab for lab in labels if lab.get("sid") in set(target_sids)]
        if len(filtered) != len(labels):
            print(f"[BATCH {bi}] filtered out {len(labels)-len(filtered)} non-target labels")

        # 欠落チェック
        got_sids = {lab.get("sid") for lab in filtered}
        exp_set = set(target_sids)
        missing_here = [sid for sid in target_sids if sid not in got_sids]
        if missing_here:
            print(f"[BATCH {bi}][WARN] missing {len(missing_here)} labels. e.g., {missing_here[:5]}")
            missing_total.extend(missing_here)

        # 順序を入力と同じに整列
        sid_to_lab = {lab["sid"]: lab for lab in filtered}
        ordered = [sid_to_lab[sid] for sid in target_sids if sid in sid_to_lab]

        # バッチの正規化後出力を保存（デバッグ）
        with open(f"{prefix}_normalized.json", "w", encoding="utf-8") as f:
            json.dump({"labels": ordered}, f, ensure_ascii=False, indent=2)

        all_labels.extend(ordered)

    # ここまでで全ターゲットの labels を獲得 → マージ
    label_map = {lab["sid"]: lab for lab in all_labels}
    merged = []
    missing_final = []
    for s in sentences:
        sid = s.get("sid")
        lab = label_map.get(sid)
        if lab is None:
            missing_final.append(sid)
            merged.append({
                **s,
                "role": None,
                "role_score": None,
                "role_reason": "missing label"
            })
        else:
            merged.append({
                **s,
                "role": lab.get("role"),
                "role_score": lab.get("role_score"),
                "role_reason": lab.get("role_reason"),
            })

    # 生の全ラベルも保存（元コード互換）
    role_raw_path = lecture_dir / "role_selections_raw.json"
    with open(role_raw_path, "w", encoding="utf-8") as f:
        json.dump({"labels": all_labels}, f, ensure_ascii=False, indent=2)

    out_path = lecture_dir / "sentences_with_roles.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[RESULT] merged {len(merged)} sentences -> {out_path.name}")
    if missing_total:
        print(f"[WARN] missing in batches: {len(missing_total)} (unique: {len(set(missing_total))})")
    if missing_final:
        print(f"[WARN] labels missing in final merge: {len(missing_final)} e.g., {missing_final[:5]}")

    print(f"⏰Classified roles: {time.time() - start_time_role_classification:.2f} seconds.")

    json_data_as_text_sentences = json.dumps(projected, ensure_ascii=False, indent=2)

    print("Waiting for response from Gemini API...")
    response_role_classification = model_json.generate_content([
        instr_role_classification,
        json_data_as_text_sentences,
        "Using the JSON data provided above, follow the instructions and return the result in JSON format."
    ])

    debug_dump_response(response_role_classification, lecture_dir / "role_selections")

    print("saving response...")
    out_role_classification = json.loads(response_role_classification.text)

    with open(lecture_dir / "role_selections_raw.json", "w", encoding="utf-8") as f:
        json.dump(out_role_classification, f, ensure_ascii=False, indent=2)

    labels = out_role_classification.get("labels", [])
    label_map = {lab["sid"]: lab for lab in labels}

    merged = []
    missing = []
    for s in sentences:
        sid = s.get("sid")
        lab = label_map.get(sid)
        if lab is None:
            missing.append(sid)
            merged.append({
                **s,
                "role": None,
                "role_score": None,
                "role_reason": "missing label"
            })
        else:
            merged.append({
                **s,
                "role": lab.get("role"),
                "role_score": lab.get("role_score"),
                "role_reason": lab.get("role_reason"),
            })

    sentence_sids = {s["sid"] for s in sentences}
    extra = [sid for sid in label_map.keys() if sid not in sentence_sids]

    with open(lecture_dir / "sentences_with_roles.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"merged {len(merged)} sentences -> sentences_with_roles.json")
    if missing:
        print(f"[WARN] labels missing for {len(missing)} sid(s). e.g., {missing[:5]}")
    if extra:
        print(f"[WARN] labels contain {len(extra)} extra sid(s). e.g., {extra[:5]}")

    end_time_role_classification = time.time()
    elapsed_time_role_classification = end_time_role_classification - start_time_role_classification
    print(f"⏰Classified roles: {elapsed_time_role_classification:.2f} seconds.") 


    # roleと文章のレビュー
    print("\n### Role and Sentence Review ###")
    start_time_role_and_sentence_review = time.time()

    with open("prompts/role_and_sentence_review.txt", "r", encoding="utf-8") as f:
        instr_role_and_sentence_review = f.read()

    with open(lecture_dir / "sentences_with_roles.json", "r", encoding="utf-8") as f:
        sentences_with_role = json.load(f)

    json_data_as_text_sentences_with_role = json.dumps(sentences_with_role, ensure_ascii=False, indent=2)

    print("Waiting for response from Gemini API...")
    response_role_and_sentence_review = model_json.generate_content([
        instr_role_and_sentence_review,
        json_data_as_text_sentences_with_role,
        "Using the JSON data provided above, follow the instructions and return the result in JSON format.",
    ])

    print("saving response...")
    out_role_and_sentence_review = json.loads(response_role_and_sentence_review.text)

    with open(lecture_dir / "reviewed_roles_and_sentences_raw.json", "w", encoding="utf-8") as f:
        json.dump(out_role_and_sentence_review, f, ensure_ascii=False, indent=2)

    with open(lecture_dir / "sentences_with_roles.json", "r", encoding="utf-8") as f:
        sentences = json.load(f)   # [{sid, text, start, end, role, ...}]

    labels = out_role_and_sentence_review.get("labels", [])
    label_map = {lab["sid"]: lab for lab in labels}

    final_rows = []
    missing = []
    extra = set(label_map.keys())
    changed_roles = 0
    changed_texts = 0

    for s in sentences:
        sid = s.get("sid")
        lab = label_map.get(sid)
        if not lab:
            missing.append(sid)
            role_final = s.get("role")
            text_final = s.get("text")
        else:
            extra.discard(sid)
            old_role = s.get("role")
            new_role = lab.get("new_role") or old_role
            role_final = new_role
            if new_role != old_role:
                changed_roles += 1

            modified = lab.get("modified", None)
            if modified is not None:
                text_final = modified
                changed_texts += 1
            else:
                text_final = s.get("text")

        final_rows.append({
            "sid": sid,
            "text": text_final,
            "start": s.get("start"),
            "end": s.get("end"),
            "role": role_final,
        })

    with open(lecture_dir / "sentences_final.json", "w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(final_rows)} rows -> sentences_final.json")
    print(f"Role changes: {changed_roles}, Text modified: {changed_texts}")
    if missing:
        print(f"[WARN] review missing for {len(missing)} sid(s). e.g., {missing[:5]}")
    if extra:
        print(f"[WARN] review has {len(extra)} extra sid(s). e.g., {list(extra)[:5]}")

    end_time_role_and_sentence_review = time.time()
    elapsed_time_role_and_sentence_review = end_time_role_and_sentence_review - start_time_role_and_sentence_review
    print(f"⏰Reviewed roles and sentences: {elapsed_time_role_and_sentence_review:.2f} seconds.")


    # トピックを選出
    print("\n### Topic Extraction ###")
    start_time_topic_extraction = time.time()

    with open("prompts/topic_extraction.txt", "r", encoding="utf-8") as f:
        instr_topic_extraction = f.read()

    with open(lecture_dir / "sentences_final.json", "r", encoding="utf-8") as f:
        sentences_final = json.load(f)

    json_data_as_text_sentences_final = json.dumps(sentences_final, ensure_ascii=False, indent=2)

    print("Waiting for response from Gemini API...")
    response_extract_topic = model_text.generate_content([
        instr_topic_extraction,
        json_data_as_text_sentences_final,
        "Using the JSON data provided above, follow the instructions and return the result in plain text.",
    ])

    print("saving response...")
    with open(lecture_dir / "topics.txt", "w", encoding="utf-8") as f:
        f.write(response_extract_topic.text)

    end_time_topic_extraction = time.time()
    elapsed_time_topic_extraction = end_time_topic_extraction - start_time_topic_extraction
    print(f"⏰Extracted topic: {elapsed_time_topic_extraction:.2f} seconds.")

    print("\n✅All tasks of TOPIC EXTRACTION completed.")


def main():
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)

    gemini_model_json = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    gemini_model_text = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "text/plain",
        },
    )

    ROOT = Path(__file__).resolve().parent
    LECTURE_DIR = ROOT / "lectures/2025-10-07-21-27-43-0700"

    topic_extraction(gemini_model_json, gemini_model_text, LECTURE_DIR)

if __name__ == "__main__":
    main()