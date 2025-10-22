import os, time, sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

from scripts.lecture_audio_to_text import lecture_audio_to_text
from scripts.topic_extraction_for_long_audio import topic_extraction_for_long_audio
from scripts.generate_topic_details import generate_topic_details
from scripts.generate_fun_facts import generate_fun_facts

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".wma", ".aiff"}

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEN_MODEL = "gemini-2.5-flash"

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
    if google_search > 0:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    return types.GenerateContentConfig(**kwargs)

def make_lecture_dir():
    ROOT = Path(__file__).resolve().parent
    LECTURES_DIR = ROOT / "lectures"
    LECTURES_DIR.mkdir(exist_ok=True)

    local_with_tz = datetime.now().astimezone().strftime("%Y-%m-%d-%H-%M-%S%z")
    LECTURE_DIR = LECTURES_DIR / local_with_tz
    LECTURE_DIR.mkdir()

    return LECTURE_DIR

def list_audio_files(dirpath: Path):
    files = []
    for p in dirpath.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            try:
                stat = p.stat()
                files.append((p, stat.st_size, stat.st_mtime))
            except FileNotFoundError:
                # copy in progress or removed between listing/stat
                pass
    return files

def human_size(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def stable_files(dirpath: Path, settle_seconds=3.0):
    snapshot1 = {p: (size, mtime) for p, size, mtime in list_audio_files(dirpath)}
    time.sleep(settle_seconds)
    snapshot2 = {p: (size, mtime) for p, size, mtime in list_audio_files(dirpath)}

    stable = []
    for p, meta1 in snapshot1.items():
        meta2 = snapshot2.get(p)
        if meta2 and meta1 == meta2:
            stable.append((p, meta2[0], meta2[1]))
    return stable

def wait_for_uploads(audio_dir: Path, min_files=1, poll_interval=1.0, settle_seconds=3.0, timeout=None):
    print(f"\n📂 Upload destination: {audio_dir.resolve()}")
    print("⬆️  Please copy your audio file(s) into this folder.")
    print("   (We'll wait here; press Ctrl+C to abort.)")
    start = time.time()
    while True:
        try:
            stable = stable_files(audio_dir, settle_seconds=settle_seconds)
            if len(stable) >= min_files:
                print(f"\n✅ Detected {len(stable)} stable file(s):")
                for p, size, _ in stable:
                    print(f" - {p.name}  [{human_size(size)}]")
                while True:
                    ans = input("\nProceed with these file(s)? [Y/n/r] "
                                "(Y: continue, n: quit, r: refresh list) ").strip().lower()
                    if ans in {"", "y", "yes"}:
                        return [p for p, _, _ in stable]
                    if ans in {"n", "no", "q", "quit"}:
                        print("💡 Aborted by user.")
                        sys.exit(0)
                    if ans in {"r", "refresh"}:
                        break  # refresh loop to re-check
            else:
                # 進行中のステータス表示
                found = list_audio_files(audio_dir)
                names = ", ".join(p.name for p, _, _ in found) or "(none yet)"
                print(f"\r⏳ Waiting for uploads... found: {names}", end="", flush=True)
                time.sleep(poll_interval)
            if timeout is not None and (time.time() - start) > timeout:
                print("\n⏱️  Timeout waiting for uploads.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted.")
            sys.exit(1)


def main():
    LECTURE_DIR = make_lecture_dir()

    AUDIO_DIR = LECTURE_DIR / "audio"
    AUDIO_DIR.mkdir()

    audio_files = wait_for_uploads(AUDIO_DIR)

    start_time_total = time.time()

    lecture_audio_to_text(audio_files[0], LECTURE_DIR)

    topic_extraction_for_long_audio(client, GEN_MODEL, config_json(), config_text(), LECTURE_DIR)

    generate_topic_details(client, GEN_MODEL, config_json(), config_text(), LECTURE_DIR)

    generate_fun_facts(client, GEN_MODEL, config_text(google_search=True), LECTURE_DIR)

    end_time_total = time.time()
    elapsed_time_total = end_time_total - start_time_total
    total_minutes = int(elapsed_time_total // 60)
    total_seconds = int(elapsed_time_total % 60)
    print(f"\n⏰⏰⏰ Total elapsed time: {total_minutes} m {total_seconds} s.")

    print("\n🎉 All tasks completed.")


if __name__ == "__main__":
    main()