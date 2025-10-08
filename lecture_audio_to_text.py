import os, json, time
import assemblyai as aai
from dotenv import load_dotenv
from pathlib import Path

def lecture_audio_to_text(audio_file, lecture_dir: Path):
    print("\n### Leccure Audio To Text ###")
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
