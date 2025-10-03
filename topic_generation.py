import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# sentencesからrole分類

model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "temperature": 0.2,
        "response_mime_type": "application/json",
    },
)

with open("prompts/role_classification.txt", "r", encoding="utf-8") as f:
    instr_role_classification = f.read()

with open("transcript_sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

ALLOWED = {"sid", "text", "start", "end", "speaker"}

projected = [{k: s.get(k) for k in ALLOWED} for s in sentences]

response = model.generate_content([
    {"role": "user", "parts": [
        {"text": instr_role_classification},
        {"inline_data": {
            "mime_type": "application/json",
            "data": json.dumps(projected).encode("utf-8")
        }},
    ]}
])

out_role_classification = json.loads(response.text)

with open("role_selections_raw.json", "w", encoding="utf-8") as f:
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
            "role_score": lab.get("score"),
            "role_reason": lab.get("reason"),
        })

sentence_sids = {s["sid"] for s in sentences}
extra = [sid for sid in label_map.keys() if sid not in sentence_sids]

with open("sentences_with_roles.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"merged {len(merged)} sentences -> sentences_with_roles.json")
if missing:
    print(f"[WARN] labels missing for {len(missing)} sid(s). e.g., {missing[:5]}")
if extra:
    print(f"[WARN] labels contain {len(extra)} extra sid(s). e.g., {extra[:5]}")


# roleと文章のレビュー

with open("prompts/role_and_sentence_review.txt", "r", encoding="utf-8") as f:
    instr_role_and_sentence_review = f.read()

with open("sentences_with_roles.json", "r", encoding="utf-8") as f:
    sentences_with_role = json.load(f)

response = model.generate_content([
    {"role": "user", "parts": [
        {"text": instr_role_and_sentence_review},
        {"inline_data": {
            "mime_type": "application/json",
            "data": json.dumps(sentences_with_role).encode("utf-8")
        }},
    ]}
])


out_role_and_sentence_review = json.loads(response.text)

with open("reviewed_roles_and_sentences_raw.json", "w", encoding="utf-8") as f:
    json.dump(out_role_and_sentence_review, f, ensure_ascii=False, indent=2)

with open("sentences_with_roles.json", "r", encoding="utf-8") as f:
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

with open("sentences_final.json", "w", encoding="utf-8") as f:
    json.dump(final_rows, f, ensure_ascii=False, indent=2)

print(f"Saved {len(final_rows)} rows -> sentences_final.json")
print(f"Role changes: {changed_roles}, Text modified: {changed_texts}")
if missing:
    print(f"[WARN] review missing for {len(missing)} sid(s). e.g., {missing[:5]}")
if extra:
    print(f"[WARN] review has {len(extra)} extra sid(s). e.g., {list(extra)[:5]}")

# トピックを選出