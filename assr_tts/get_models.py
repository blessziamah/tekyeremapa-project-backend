import json
import os

import librosa
import torch
from assr_tts import processor, stt_model
from assr_tts import stt_model
import requests
from difflib import SequenceMatcher


def get_transcription(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    # audio_stretched = librosa.effects.time_stretch(audio, rate=1.5)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        output = stt_model.generate(**inputs)

    transcript = processor.batch_decode(output, skip_special_tokens=True)[0]
    return transcript


# Usage example
# print(get_transcription("data/1.mp3"))
def get_speech(text, language="tw", speaker_id="twi_speaker_8", output_file="output.wav"):
    try:
        url = os.environ.get("TTS_URL", "https://translation-api.ghananlp.org/tts/v1/synthesize")
        if not url:
            return False, "Error occurred: TTS URL is not configured"

        headers = {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
            'Ocp-Apim-Subscription-Key': os.environ.get("TTS_API_KEY"),
        }

        data = {
            "text": text,
            "language": language,
            "speaker_id": speaker_id
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            # Save the audio content to a WAV file
            with open(output_file, "wb") as f:
                f.write(response.content)
            return True, "Success"
        else:
            return False, f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return False, f"Error occurred: {e}"


def load_word_data():
    """Load word data from the JSON file."""
    # Go up one directory level to reach the project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, 'data', 'data.json')
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_word_by_id(word_id):
    """Retrieve a word by its ID.

    Args:
        word_id (int): The ID of the word to retrieve.

    Returns:
        dict: The word data if found, None otherwise.
    """
    data = load_word_data()
    for word_data in data.get('words', []):
        if word_data['id'] == word_id:
            return word_data
    return None



def get_evaluation(audio_path, id):
    """Evaluate if the audio matches the word with the given ID.

    Args:
        audio_path (str): Path to the audio file to evaluate.
        id (int): ID of the word to compare against.

    Returns:
        dict: A dictionary containing the evaluation result, the word data, and the transcription.
    """
    word_data = get_word_by_id(int(id))
    if not word_data:
        return {"success": False, "error": f"Word with ID {id} not found"}

    try:
        # transcribe the audio
        transcription = get_transcription(audio_path)
        clean_transcription = transcription.lower().strip()
        clean_word = word_data['word'].lower().strip()

        similarity = SequenceMatcher(None, clean_transcription, clean_word).ratio()
        percentage = round(similarity * 100, 2)
        success = percentage >= 70  

        return {
            "success": True,
            "similarity_percentage": percentage,
            "passed": success,
            "expected": clean_word,
            "actual": clean_transcription,
            "word_data": word_data
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
