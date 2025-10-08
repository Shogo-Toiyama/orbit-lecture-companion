import time
from pathlib import Path

def generate_fun_facts(model_text, lecture_dir: Path):
    # topicsごとのfun factを生成
    print("\n### Fun Fact Generation ###")

    start_time_fun_facts = time.time()

    FUN_FACT_DIR = Path(lecture_dir / "fun_facts")
    FUN_FACT_DIR.mkdir()

    with open("prompts/fun_fact_generation.txt", "r", encoding="utf-8") as f:
        instr_fun_facts_generation = f.read()

    detail_files = sorted(Path(lecture_dir / "details/edited").glob("* - details.txt"))

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