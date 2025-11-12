import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

def token_report(resp):
    um = resp.usage_metadata
    prompt_tokens = um.prompt_token_count
    candidate_tokens = um.candidates_token_count
    total_tokens = um.total_token_count
    thinking_tokens = total_tokens - prompt_tokens - candidate_tokens
    return (f"TOKEN USAGE REPORT\n  ⬆️:{prompt_tokens}, 🧠: {thinking_tokens}, ⬇️: {candidate_tokens}\n  TOTAL: {total_tokens}")

def _generate_one_fun_fact(
    client, gen_model, config_text,
    instr_fun_facts_generation: str,
    detail_file: Path, fun_fact_dir: Path
):
    start_time_one_fun_fact = time.time()
    topic_name = detail_file.stem.replace(" - details", "")

    with open(detail_file, "r", encoding="utf-8") as f:
        topic_details_markdown = f.read()

    print(f"Waiting for response from Gemini API for {detail_file.name}...")
    contents = [
        instr_fun_facts_generation,
        topic_details_markdown,
        "Using the text provided above, follow the instructions and return the result in markdown text."
    ]
    response_fun_facts = client.models.generate_content(
        model = gen_model,
        contents = contents,
        config = config_text,
    )

    print("saving response...")

    with open(fun_fact_dir / f"{topic_name} - fun_fact.txt", "w", encoding="utf-8") as f:
        f.write(response_fun_facts.text)
    
    end_time_one_fun_fact = time.time()
    elapsed_time_one_fun_fact = end_time_one_fun_fact - start_time_one_fun_fact
    print(token_report(response_fun_facts))
    print(f"  --> ⏰Generated fun fact for '{topic_name}': {elapsed_time_one_fun_fact:.2f} seconds.")

def generate_fun_facts(client, gen_model, config_text, lecture_dir: Path):
    # topicsごとのfun factを生成
    print("\n### Fun Fact Generation ###")

    start_time_fun_facts = time.time()

    FUN_FACT_DIR = Path(lecture_dir / "fun_facts")
    FUN_FACT_DIR.mkdir(exist_ok=True, parents=True)

    with open(PROMPTS_DIR / "fun_fact_generation.txt", "r", encoding="utf-8") as f:
        instr_fun_facts_generation = f.read()

    detail_files = sorted(Path(lecture_dir / "details").glob("* - details.txt"))
    
    max_workers = 3
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        submit_one = partial(
            _generate_one_fun_fact,
            client, gen_model, config_text,
            instr_fun_facts_generation, fun_fact_dir = FUN_FACT_DIR
        )
        futures = {ex.submit(submit_one, detail): detail for detail in detail_files}

        for fut in as_completed(futures):
            f = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"❌ Generating fun facts failed for {f}: {e}")

    end_time_fun_facts = time.time()
    elapsed_time_fun_facts = end_time_fun_facts - start_time_fun_facts
    print(f"⏰Generated all fun facts: {elapsed_time_fun_facts:.2f} seconds.")

    print("\n✅All tasks of FUN FACT GENERATION completed.")


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
    LECTURE_DIR = ROOT / "../lectures/2025-11-11-16-38-54-0800"

    generate_fun_facts(client, GEN_MODEL, config_text(google_search=True), LECTURE_DIR)

if __name__ == "__main__":
    main()