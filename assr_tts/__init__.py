from dotenv import load_dotenv
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer, AutoModelForMaskedLM, \
    WhisperTokenizerFast

load_dotenv()

stt_model_path = "../models/akan-non-standard-tiny"
# stt_model_path_hci = "/Users/blessziamah/Achieve/tekyerema-project-backendf/models/hci_lab"

# tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny")
# tokenizer.save_pretrained(stt_model_path_hci)
processor = WhisperProcessor.from_pretrained(stt_model_path)
stt_model = WhisperForConditionalGeneration.from_pretrained(stt_model_path)


