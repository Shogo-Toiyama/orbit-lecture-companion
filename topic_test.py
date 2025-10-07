import re, json, time

out = []

with open("topics.txt", "r", encoding="utf-8", newline=None) as f:
    for l in f:
        raw = l.strip()
        if not raw:
            continue
        cleaned = re.sub(r'^\s*(\d+|[-*])\s*[\.\)\-\–\—]?\s*', '', raw)
        m = re.match(r'^\s*(\d+)', raw)
        idx = int(m.group(1)) if m else None
        out.append({"idx": idx, "title": cleaned, "raw": raw})

for item in out:
    print(item)