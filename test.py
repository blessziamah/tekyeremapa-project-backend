import requests
import wave
import io

def text_to_speech(text, language="tw", speaker_id="twi_speaker_8", output_file="output.wav"):
    try:
        url = "https://translation-api.ghananlp.org/tts/v1/synthesize"
        
        headers = {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
            'Ocp-Apim-Subscription-Key': '64ba516bcbab435da3532395718bae74',
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

# Example usage:
success, message = text_to_speech("Nnipa no retwa anwea bi mu.")
if success:
    print("Audio file saved successfully")
else:
    print(message)