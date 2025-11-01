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

def split_balanced(n_items: int, max_batch: int):
    if n_items <= 0:
        return []
    if max_batch <= 0:
        raise ValueError("max_batch must be positive")
    n_batches = math.ceil(n_items / max_batch)
    base = n_items // n_batches
    rem = n_items % n_batches
    ranges = []
    start = 0
    for i in range(n_batches):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges

def save_batches(data, batch_num: int, start: int, end: int, ctx: int, batch_dir: Path):
    n = len(data)
    main_text_chunk = data[start : end]
    ctx_bf_mt_chunk = data[max(0, start - ctx): start]
    ctx_af_mt_chunk = data[end : min(n, end + ctx)]
    obj = [
        {
            "context_before_main_text": ctx_bf_mt_chunk,
            "main_text": main_text_chunk,
            "context_after_main_text": ctx_af_mt_chunk,
        }
    ]
    with open(batch_dir / f"batch_{batch_num:02d}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2)

async def run_one_role_classification(client, gen_model, config_json, prompt, batch_path: Path):
    start_time_one_role_classification = time.time()
    batch_dir = batch_path.parent
    contents = [
        prompt,
        batch_path.read_text(encoding="utf-8"),
        "Using the JSON data provided above, follow the instructions and return the result in JSON format.",
    ]
    try:
        print(f"Waiting for response {batch_path.name} from Gemini API...")
        resp = await client.aio.models.generate_content(
            model = gen_model,
            contents = contents,
            config = config_json
        )
        result_path = batch_dir / f"role_classifications_batch.json"
        result_path.write_text(resp.text, encoding="utf-8")
        print(f"✅ Saved {result_path.name}")
        end_time_one_role_classification = time.time()
        elapsed_time_one_role_classification = end_time_one_role_classification - start_time_one_role_classification
        print(f"⏰One Role Classification of {batch_path.name}: {elapsed_time_one_role_classification:.2f} seconds.")
        return resp.text
    except Exception as e:
        print(f"❌ Error in {batch_dir.name}: {e}")
        return False

async def run_all_role_classification(client, gen_model, config_json, batches_dir: Path):
    prompt = Path(PROMPTS_DIR / "role_classification.txt").read_text(encoding="utf-8")
    sem = asyncio.Semaphore(6)
    batch_files = sorted((batches_dir).glob("batch_*/batch_*.json"))
    print(f"Found {len(batch_files)} batches under {batches_dir}")
    async def sem_task(batch_file: Path):
        async with sem:
            out_file = batch_file.parent / "role_classifications_batch.json"
            if out_file.exists():
                print(f"⏭️  Skip (exists) {out_file.relative_to(Path.cwd())}")
                return True
            return await run_one_role_classification(
                client, gen_model, config_json, prompt, batch_file
            )

    results = await asyncio.gather(*(sem_task(f) for f in batch_files))
    success = sum(1 for r in results if r)
    print(f"\n✅ Completed {success}/{len(batch_files)} batches")

def _strip_code_fence(text: str) -> str:
    if text.lstrip().startswith("```"):
        lines = [ln.rstrip("\n") for ln in text.splitlines()]
        # 先頭の```を落とす
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # 末尾の```を落とす
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text

def _sid_to_int(sid: Optional[str]) -> Optional[int]:
    if not sid:
        return None
    m = SID_NUM.fullmatch(sid)
    return int(m.group(1)) if m else None

def merge_role_classifications(lecture_dir: Path, strict_continuity: bool = True):
    batches_dir = lecture_dir / "role_batches"
    files = sorted(batches_dir.glob("batch_*/role_classifications_batch.json"))
    if not files:
        raise FileNotFoundError(f"No role_classifications_batch.json found under {batches_dir}")

    merged_labels = []
    seen_sids = set()

    prev_sid_num = None
    prev_sid = None

    total_files = 0
    total_items = 0
    skipped_dups = 0

    for f in files:
        text = _strip_code_fence(f.read_text(encoding="utf-8"))
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse failed in {f}: {e}") from e

        labels = obj.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"{f} does not contain a 'labels' array")

        total_files += 1
        total_items += len(labels)

        for item in labels:
            sid = item.get("sid")
            # 重複 sid はスキップ（コンテキスト重なり対策）
            if sid in seen_sids:
                skipped_dups += 1
                continue

            # 連番チェック（任意）
            if strict_continuity:
                cur = _sid_to_int(sid)
                # ファイル頭同士の跨ぎもチェックできる
                if prev_sid_num is not None and cur is not None:
                    expected = prev_sid_num + 1
                    if cur != expected:
                        raise AssertionError(
                            f"SID continuity broken: expected s{expected:06d} after {prev_sid}, got {sid} in {f}"
                        )
                if cur is not None:
                    prev_sid_num = cur
                    prev_sid = sid

            merged_labels.append(item)
            seen_sids.add(sid)

    out_path = lecture_dir / "role_classifications.json"
    out_path.write_text(
        json.dumps({"labels": merged_labels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"📦 Merged {total_files} files, {total_items} items → {len(merged_labels)} unique items "
        f"(skipped {skipped_dups} dups) -> {out_path.name}"
    )
    return out_path

def sentence_review(client, gen_model, config_json, lecture_dir: Path):
    # sentencesのReview
    print("\n### Sentence Review ###")
    start_time_sentence_review = time.time()

    REVIEWED_DIR = lecture_dir / "reviewed"
    REVIEWED_DIR.mkdir(exist_ok=True)

    instr_sentence_review = Path(PROMPTS_DIR / "sentence_review.txt").read_text(encoding="utf-8")

    with open(lecture_dir / "transcript_sentences.json", "r", encoding="utf-8") as f:
        sentences = json.load(f)

    ALLOWED = ["sid", "text", "confidence"]
    projected_sentences = [{k: s.get(k) for k in ALLOWED} for s in sentences]
    low_confidence_sid = [s.get("sid") for s in sentences if s.get("confidence") < 0.9]
    print("Low Confident Sentences: ", len(low_confidence_sid))

    payload = {
        "task": "Sentence Review",
        "instruction": instr_sentence_review,
        "data": {
            "original_sentences": projected_sentences,
            "low_confidence_sid": low_confidence_sid
        }
    }

    contents = [
        "This is very important task, but I am sure that you will do this well, because you are the best data processer. Read the JSON and follow the instructions carefully.",
        json.dumps(payload, ensure_ascii=False)
    ]

    print("Waiting for response from Gemini API...")
    response_sentence_review = client.models.generate_content(
        model = gen_model,
        contents = contents,
        config = config_json,
    )

    print("saving response...")
    raw_text = response_sentence_review.text
    clean_text = _strip_code_fence(raw_text).strip()
    out_review_sentence = json.loads(clean_text)

    with open (lecture_dir / "reviewed/reviewed_sentences_raw.json", "w", encoding="utf-8") as f:
        json.dump(out_review_sentence, f, ensure_ascii=False, indent=2)

    sentence_reviewed_list = out_review_sentence.get("results")

    mods = {}

    for r in sentence_reviewed_list:
        sid = r.get("sid")
        modified = r.get("modified")
        if sid and isinstance(modified, str) and modified.strip():
            mods[sid] = modified.strip()
    
    reviewed_sentences = []

    for s in sentences:
        sid = s.get("sid")
        if sid in mods:
            new_s = dict(s)
            new_s["text"] = mods[sid]
            reviewed_sentences.append(new_s)
        else:
            reviewed_sentences.append(s)

    with open (lecture_dir / "reviewed_sentences.json", "w", encoding="utf-8") as f:
        json.dump(reviewed_sentences, f, ensure_ascii=False, indent=2)

    end_time_sentence_review = time.time()
    elapsed_time_sentence_review = end_time_sentence_review - start_time_sentence_review
    print(f"⏰Sentence Review: {elapsed_time_sentence_review:.2f} seconds.") 


def role_classification(client, gen_model, config_json, lecture_dir: Path, max_batch_size: int = 300, ctx: int = 10):
    # sentencesからrole分類
    print("\n### Role Classification ###")
    start_time_role_classification = time.time()

    with open(lecture_dir / "reviewed_sentences.json", "r", encoding="utf-8") as f:
        sentences = json.load(f)

    ALLOWED = ["sid", "text", "start", "end", "speaker"]

    projected = [{k: s.get(k) for k in ALLOWED} for s in sentences]

    print("\n --> Separate Json to batches")
    n = len(projected)
    print(f"[INFO] sentences: {n}")
    ranges = split_balanced(n, max_batch_size)
    print(f"[INFO] {len(ranges)} batches: {ranges}")

    batches_dir = lecture_dir / "role_batches"
    batches_dir.mkdir(exist_ok=True)

    for i, (start, end) in enumerate(ranges):
        batch_num = i+1
        batch_dir = batches_dir / f"batch_{batch_num:02d}"
        batch_dir.mkdir(exist_ok=True, parents=True)
        save_batches(projected, batch_num, start, end, ctx, batch_dir)
    
    asyncio.run(run_all_role_classification(client, gen_model, config_json, batches_dir))
    merge_role_classifications(lecture_dir)

    with open(lecture_dir / "role_classifications.json", "r", encoding="utf-8") as f:
        out_role_classification = json.load(f)
    
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


def role_review(client, gen_model, config_json, lecture_dir: Path):
    # roleと文章のレビュー
    print("\n### Role Review ###")
    start_time_role_review = time.time()

    REVIEWED_DIR = lecture_dir / "reviewed"
    REVIEWED_DIR.mkdir(exist_ok=True)

    instr_role_review = Path(PROMPTS_DIR / "role_review.txt").read_text(encoding="utf-8")

    with open(lecture_dir / "sentences_with_roles.json", "r", encoding="utf-8") as f:
        sentences_with_role = json.load(f)

    ALLOWED = ["sid", "text", "role", "role_score"]
    projected = [{k: s.get(k) for k in ALLOWED} for s in sentences_with_role]

    low_confidence_sid = [s.get("sid") for s in sentences_with_role if s.get("role_score") < 0.9]
    print("Low Confident Roles: ", len(low_confidence_sid))

    payload = {
        "task": "Role Review",
        "instruction": instr_role_review,
        "data": {
            "sentences_with_role": projected,
            "low_confidence_sid": low_confidence_sid,
        }
    }

    contents = [
        "This is very important task, but I am sure that you will do this well, because you are the best data processer. Read the JSON and follow the instructions carefully.",
        json.dumps(payload, ensure_ascii=False)
    ]

    print("Waiting for response from Gemini API...")
    response_role_review = client.models.generate_content(
        model = gen_model,
        contents = contents,
        config = config_json,
    )

    print("saving response...")
    raw_text = response_role_review.text
    clean_text = _strip_code_fence(raw_text).strip()
    out_role_review = json.loads(clean_text)

    with open(lecture_dir / "reviewed/reviewed_roles_raw.json", "w", encoding="utf-8") as f:
        json.dump(out_role_review, f, ensure_ascii=False, indent=2)

    reviewed_role_list = out_role_review.get("results")

    new_roles = {}

    for r in reviewed_role_list:
        sid = r.get("sid")
        changed = r.get("new_role")
        if sid and isinstance(changed, str) and changed.strip():
            new_roles[sid] = changed.strip()

    reviewed_sentences = []
    for s in sentences_with_role:
        sid = s.get("sid")
        if sid in new_roles:
            new_s = dict(s)
            new_s["role"] = new_roles[sid]
            reviewed_sentences.append(new_s)
        else:
            reviewed_sentences.append(s)

    KEYS = ["sid", "text", "start", "end", "speaker", "role"]
    sentences_final = [{k: r.get(k) for k in KEYS} for r in reviewed_sentences]

    with open(lecture_dir / "sentences_final.json", "w", encoding="utf-8") as f:
        json.dump(sentences_final, f, ensure_ascii=False, indent=2)
    
    end_time_role_review = time.time()
    elapsed_time_role_review = end_time_role_review - start_time_role_review
    print(f"⏰Reviewed roles: {elapsed_time_role_review:.2f} seconds.")


# def role_and_sentence_review(client, gen_model, config_json, lecture_dir: Path):
# # roleと文章のレビュー
#     print("\n### Role and Sentence Review ###")
#     start_time_role_and_sentence_review = time.time()

#     instr_role_and_sentence_review = Path(PROMPTS_DIR / "role_and_sentence_review.txt").read_text(encoding="utf-8")

#     with open(lecture_dir / "sentences_with_roles.json", "r", encoding="utf-8") as f:
#         sentences_with_role = json.load(f)
    
#     sentences_with_role_as_text = json.dumps(sentences_with_role, ensure_ascii=False, indent=2)

#     print("Waiting for response from Gemini API...")
#     contents = [
#         instr_role_and_sentence_review,
#         sentences_with_role_as_text,
#         "Using the JSON data provided above, follow the instructions and return the result in JSON format.",
#     ]
#     response_role_and_sentence_review = client.models.generate_content(
#         model = gen_model,
#         contents = contents,
#         config = config_json,
#     )

#     print("saving response...")
#     raw_text = response_role_and_sentence_review.text
#     clean_text = _strip_code_fence(raw_text).strip()
#     out_role_and_sentence_review = json.loads(clean_text)

#     # out_role_and_sentence_review = {"changes": [], "fixes": []}

#     with open(lecture_dir / "reviewed_roles_and_sentences_raw.json", "w", encoding="utf-8") as f:
#         json.dump(out_role_and_sentence_review, f, ensure_ascii=False, indent=2)

#     sentences = sentences_with_role

#     changes  = out_role_and_sentence_review.get("changes", [])
#     fixes    = out_role_and_sentence_review.get("fixes", [])

#     change_map = {c["sid"]: c.get("new_role") for c in changes if "sid" in c and c.get("new_role")}
#     fix_map    = {f["sid"]: f.get("modified") for f in fixes if "sid" in f and f.get("modified")}

#     # 3) sentences に反映して最終出力を作る
#     final_rows = []
#     changed_roles = 0
#     changed_texts = 0

#     for s in sentences:
#         sid = s.get("sid")

#         # 役割は基本オリジナル。changesにあるsidのみ上書き
#         role_original = s.get("role")
#         role_final = change_map.get(sid, role_original)
#         if role_final != role_original:
#             changed_roles += 1

#         # テキスト修正（あるものだけ）
#         text_original = s.get("text")
#         text_final = fix_map.get(sid, text_original)
#         if text_final != text_original:
#             changed_texts += 1

#         final_rows.append({
#             "sid": sid,
#             "text": text_final,
#             "start": s.get("start"),
#             "end": s.get("end"),
#             "speaker": s.get("speaker"),
#             "confidence": s.get("confidence"),
#             "role": role_final,
#         })    

#     with open(lecture_dir / "sentences_final.json", "w", encoding="utf-8") as f:
#         json.dump(final_rows, f, ensure_ascii=False, indent=2)

#     print(f"Saved {len(final_rows)} rows -> sentences_final.json")
#     print(f"Role changes: {changed_roles}, Text modified: {changed_texts}")
    
#     end_time_role_and_sentence_review = time.time()
#     elapsed_time_role_and_sentence_review = end_time_role_and_sentence_review - start_time_role_and_sentence_review
#     print(f"⏰Reviewed roles and sentences: {elapsed_time_role_and_sentence_review:.2f} seconds.")

def topic_extraction(client, gen_model, config_text, lecture_dir: Path):
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


def topic_extraction_for_long_audio(client, gen_model, gen_model_lite, config_json, config_text, lecture_dir: Path, max_batch_size: int = 300, ctx: int = 10):

    sentence_review(client, gen_model, config_json, lecture_dir)
    
    role_classification(client, gen_model_lite, config_json, lecture_dir, max_batch_size, ctx)

    role_review(client, gen_model, config_json, lecture_dir)

    # role_and_sentence_review(client, gen_model, config_json, lecture_dir)

    topic_extraction(client, gen_model, config_text, lecture_dir)

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

    topic_extraction_for_long_audio(client, flash, flash_lite, config_json(), config_text(), LECTURE_DIR)

    # sentence_review(client, flash, config_json(), LECTURE_DIR)
    
    # role_classification(client, flash_lite, config_json(), LECTURE_DIR, 300, 10)

    # role_review(client, flash, config_json(), LECTURE_DIR)

    # topic_extraction(client, flash, config_text(), LECTURE_DIR)

if __name__ == "__main__":
    main()