import whisper
import json

model = whisper.load_model("tiny")  # "tiny", "base", "small", "medium", "large"
file_path = "./audio/2025 Engineering 1 TR - Basic Training Coding 1 workshop - 6_24_2025.mp3"
result = model.transcribe(file_path)

with open("transcription.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)