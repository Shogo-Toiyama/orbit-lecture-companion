import os, json, time
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# sentencesからrole分類
print("### Role Classification ###")
start_time_role_classification = time.time()

model_json = genai.GenerativeModel(
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

json_data_as_text_sentences = json.dumps(projected, ensure_ascii=False, indent=2)

print("Waiting for response from Gemini API...")
response_role_classification = model_json.generate_content([
    instr_role_classification,
    json_data_as_text_sentences,
    "Using the JSON data provided above, follow the instructions and return the result in JSON format."
])

print("saving response...")
out_role_classification = json.loads(response_role_classification.text)

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

end_time_role_classification = time.time()
elapsed_time_role_classification = end_time_role_classification - start_time_role_classification
print(f"⏰Classified roles: {elapsed_time_role_classification:.2f} seconds.") 


# roleと文章のレビュー
print("\n### Role and Sentence Review ###")
start_time_role_and_sentence_review = time.time()

with open("prompts/role_and_sentence_review.txt", "r", encoding="utf-8") as f:
    instr_role_and_sentence_review = f.read()

with open("sentences_with_roles.json", "r", encoding="utf-8") as f:
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

end_time_role_and_sentence_review = time.time()
elapsed_time_role_and_sentence_review = end_time_role_and_sentence_review - start_time_role_and_sentence_review
print(f"⏰Reviewed roles and sentences: {elapsed_time_role_and_sentence_review:.2f} seconds.")


# トピックを選出
print("\n### Topic Extraction ###")
start_time_topic_extraction = time.time()

model_text = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "temperature": 0.2,
        "response_mime_type": "text/plain",
    },
)

with open("prompts/topic_extraction.txt", "r", encoding="utf-8") as f:
    instr_topic_extraction = f.read()

with open("sentences_final.json", "r", encoding="utf-8") as f:
    sentences_final = json.load(f)

json_data_as_text_sentences_final = json.dumps(sentences_final, ensure_ascii=False, indent=2)

print("Waiting for response from Gemini API...")
response_extract_topic = model_text.generate_content([
    instr_topic_extraction,
    json_data_as_text_sentences_final,
    "Using the JSON data provided above, follow the instructions and return the result in plain text.",
])

print("saving response...")
with open("topics.txt", "w", encoding="utf-8") as f:
    f.write(response_extract_topic.text)

end_time_topic_extraction = time.time()
elapsed_time_topic_extraction = end_time_topic_extraction - start_time_topic_extraction
print(f"⏰Extracted topic: {elapsed_time_topic_extraction:.2f} seconds.")

print("\n✅All tasks of TOPIC EXTRACTION completed.")