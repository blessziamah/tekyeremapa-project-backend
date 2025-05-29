from faster_whisper import WhisperModel
import os
import torch

# Construct the absolute path to your local model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/sumaya")  # Make sure this folder has 'model.bin', 'config.json', etc.
print(MODEL_PATH)

# Choose the device
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperModel(
    model_size_or_path=MODEL_PATH,  # Now correctly interpreted as a local path
    device=device,
    compute_type="int8",  # Use "float16" or "float32" for better accuracy if supported
    local_files_only=True
)

# def transcribe_with_whisper(audio_data):
#     segments, _ = model.transcribe(
#         audio_data,
#         beam_size=5,
#         language='en',
#         word_timestamps=False,
#         vad_filter=True
#     )
#
#     return " ".join(segment.text for segment in segments)
#
# print(transcribe_with_whisper("./data/1.mp3"))

# from transformers import WhisperTokenizer
#
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large")
# tokenizer.save_pretrained("./models/hci_lab_small/fast")
