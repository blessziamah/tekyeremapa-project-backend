from dotenv import load_dotenv
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer, AutoModelForMaskedLM, \
    WhisperTokenizerFast

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
stt_model_path = os.path.join(project_root, "models", "akan-non-standard-large")

# tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-large-v2")
# tokenizer.save_pretrained(stt_model_path)

processor = WhisperProcessor.from_pretrained(stt_model_path, local_files_only=True)
stt_model = WhisperForConditionalGeneration.from_pretrained(stt_model_path, local_files_only=True)
