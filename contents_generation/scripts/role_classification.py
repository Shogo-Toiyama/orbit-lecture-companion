import json, re, time, math, asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


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

def role_classification_draft(client, gen_model, config_json, lecture_dir: Path, max_batch_size: int = 300, ctx: int = 10):
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


def role_classification(client, gen_model, gen_model_lite, config_json, lecture_dir: Path, max_batch_size: int = 350, ctx: int = 10):

    role_classification_draft(client, gen_model_lite, config_json, lecture_dir, max_batch_size, ctx)

    role_review(client, gen_model, config_json, lecture_dir)

    print("\n✅All tasks of ROLE CLASSIFICATION completed.")