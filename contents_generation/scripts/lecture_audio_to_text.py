import os, json, time
import assemblyai as aai
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

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

def speach_to_text(audio_file, lecture_dir: Path):
    print("\n### Lecture Audio To Text ###")
    start_time_audio_to_text = time.time()

    load_dotenv()
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    aai_config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal,
        speaker_labels=True
    )

    print("Waiting for response from AssemblyAI API...")
    transcript = aai.Transcriber(config=aai_config).transcribe(str(audio_file))

    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    print("saving response...")
    with open(lecture_dir / "transcript_raw.json", "w", encoding="utf-8") as f:
        json.dump(transcript.json_response, f, ensure_ascii=False, indent=2)

    sentences = transcript.get_sentences()

    def sentence_to_dict(s, idx):
        return {
            "sid": f"s{idx:06d}",
            "text": getattr(s, "text", None),
            "start": getattr(s, "start", None),
            "end": getattr(s, "end", None),
            "speaker": getattr(s, "speaker", None),
            "confidence": getattr(s, "confidence", None),
        }

    data = [sentence_to_dict(s, idx) for idx, s in enumerate(sentences, start=1)]
    with open(lecture_dir / "transcript_sentences.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    end_time_audio_to_text = time.time()
    elapsed_time_audio_to_text = end_time_audio_to_text - start_time_audio_to_text
    print(f"⏰Transcribed audio to text: {elapsed_time_audio_to_text:.2f} seconds.")


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

def lecture_audio_to_text(audio_file, lecture_dir: Path, client, gen_model, config_json):

    speach_to_text(audio_file, lecture_dir)
    sentence_review(client, gen_model, config_json, lecture_dir)
    

def main():
    ROOT = Path(__file__).resolve().parent
    LECTURE_DIR = ROOT / "../lectures/2025-10-27-16-29-28-0700"
    lecture_audio_to_text(LECTURE_DIR)

if __name__ == "__main__":
    main()