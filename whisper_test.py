from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# base_folder = "lectures"
# subfolders = ["audio", "transcription", "summary"]
# session_folder = os.path.join(base_folder, timestamp)
# for sub in subfolders:
#     full_path = os.path.join(session_folder, sub)
#     os.makedirs(full_path, exist_ok=True)
#     print(f"Created: {full_path}")

load_dotenv()
openai_api_key = os.getenv("SHOGO_S_OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
file_path = "audio/AI_and_ethics_10_copy.mp3"

# with open(file_path, "rb") as f1:
#     transcription_with_whisper1 = client.audio.transcriptions.create(
#         model="whisper-1",
#         file=f1,
#         response_format="verbose_json",
#     )


# with open("transcription/taiju_no_randoom_audio_2025-04-03_08_12_10_with_whisper1.json", "w", encoding="utf-8") as f:
#     json.dump(transcription_with_whisper1.model_dump(), f, ensure_ascii=False, indent=2)


with open(file_path, "rb") as f2:
    transcription_with_gpt4o = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=f2,
        response_format="text",
    )

output_txt_path = "transcription/AI_and_ethics_10_copy_with_gpt4o.txt"
os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(transcription_with_gpt4o)


# with open("transcription/taiju_no_randoom_audio_2025-04-03_08_12_10_with_gpt4o.json", "w", encoding="utf-8") as f:
#     json.dump(transcription_with_gpt4o.model_dump(), f, ensure_ascii=False, indent=2)