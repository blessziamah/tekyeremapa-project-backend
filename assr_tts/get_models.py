import os

import librosa
import torch
from assr_tts import processor, stt_model
from assr_tts import stt_model
import requests


def get_transcription(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        output = stt_model.generate(**inputs)

    transcript = processor.batch_decode(output, skip_special_tokens=True)[0]
    return transcript


# Usage example
print(get_transcription("../data/1.mp3"))

def get_speech(text, language="tw", speaker_id="twi_speaker_8", output_file="output.wav"):
    try:
        url = os.environ.get("TTS_API_URL")

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

# Usage example
# get_speech("Wo ho te sen")
