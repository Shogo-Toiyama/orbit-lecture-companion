import os, time, re
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# topicsごとのfun factを生成
print("\n### Fun Fact Generation ###")

model_text = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "temperature": 0.2,
        "response_mime_type": "text/plain",
    },
)

start_time_fun_facts = time.time()

FUN_FACT_DIR = Path("fun_facts")
FUN_FACT_DIR.mkdir(exist_ok=True)

with open("prompts/fun_fact_generation.txt", "r", encoding="utf-8") as f:
    instr_fun_facts_generation = f.read()

detail_files = sorted(Path("details/edited").glob("* - details.txt"))

for i in range(len(detail_files)):
    start_time_one_fun_fact = time.time()
    detail_file = detail_files[i]
    topic_name = detail_file.stem.replace(" - details", "")

    with open(detail_file, "r", encoding="utf-8") as f:
        topic_details_markdown = f.read()

    print("Waiting for response from Gemini API...")
    response_fun_facts = model_text.generate_content([
        instr_fun_facts_generation,
        topic_details_markdown,
        "Using the text provided above, follow the instructions and return the result in markdown text."
    ])

    print("saving response...")

    with open(FUN_FACT_DIR / f"{topic_name} - fun_fact.txt", "w", encoding="utf-8") as f:
        f.write(response_fun_facts.text)
    
    end_time_one_fun_fact = time.time()
    elapsed_time_one_fun_fact = end_time_one_fun_fact - start_time_one_fun_fact
    print(f"  --> ⏰Generated fun fact for '{topic_name}': {elapsed_time_one_fun_fact:.2f} seconds.")

end_time_fun_facts = time.time()
elapsed_time_fun_facts = end_time_fun_facts - start_time_fun_facts
print(f"⏰Generated all fun facts: {elapsed_time_fun_facts:.2f} seconds.")

print("\n✅All tasks of FUN FACT GENERATION completed.")