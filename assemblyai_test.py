from dotenv import load_dotenv
import os
import json
import assemblyai as aai

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

audio_file = "./audio/AI_and_ethics_10_copy.mp3"

config = aai.TranscriptionConfig(
    speech_model=aai.SpeechModel.universal,
    speaker_labels=True
)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
    raise RuntimeError(f"Transcription failed: {transcript.error}")

# ---- 1) 生レスポンス（全フィールド） ----
with open("transcript_raw.json", "w", encoding="utf-8") as f:
    json.dump(transcript.json_response, f, ensure_ascii=False, indent=2)

# ---- 2) 読みやすいビュー（Sentence/Paragraph）を手動dict化して保存 ----
sentences = transcript.get_sentences()
paragraphs = transcript.get_paragraphs()
utterances = transcript.utterances()

def word_to_dict(w):
    # SDKのWordオブジェクトから必要なフィールドだけ取り出す
    return {
        "text": getattr(w, "text", None),
        "start": getattr(w, "start", None),
        "end": getattr(w, "end", None),
        "confidence": getattr(w, "confidence", None),
        "speaker": getattr(w, "speaker", None),
    }

def sentence_to_dict(s):
    return {
        "text": getattr(s, "text", None),
        "start": getattr(s, "start", None),
        "end": getattr(s, "end", None),
        "confidence": getattr(s, "confidence", None),
        "speaker": getattr(s, "speaker", None),
        "words": [word_to_dict(w) for w in getattr(s, "words", [])],
    }

def paragraph_to_dict(p):
    return {
        "text": getattr(p, "text", None),
        "start": getattr(p, "start", None),
        "end": getattr(p, "end", None),
        "confidence": getattr(p, "confidence", None),
        "speaker": getattr(p, "speaker", None),
        "sentences": [sentence_to_dict(s) for s in getattr(p, "sentences", [])],
    }

with open("transcript_sentences.json", "w", encoding="utf-8") as f:
    json.dump([sentence_to_dict(s) for s in sentences], f, ensure_ascii=False, indent=2)

with open("transcript_paragraphs.json", "w", encoding="utf-8") as f:
    json.dump([paragraph_to_dict(p) for p in paragraphs], f, ensure_ascii=False, indent=2)

print("Saved: transcript_raw.json / transcript_sentences.json / transcript_paragraphs.json")
