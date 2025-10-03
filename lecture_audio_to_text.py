from dotenv import load_dotenv
import os
import json
import assemblyai as aai

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

#あとで日付とかでフォルダを決めてその中のオーディオファイルを読み込む！
audio_file = "./audio/AI_and_ethics_10_copy.mp3"

config = aai.TranscriptionConfig(
    speech_model=aai.SpeechModel.universal,
    speaker_labels=True
)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
    raise RuntimeError(f"Transcription failed: {transcript.error}")

with open("transcript_raw.json", "w", encoding="utf-8") as f:
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
with open("transcript_sentences.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

