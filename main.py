import os
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body, Path as FastAPIPath
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from assr_tts.get_models import get_speech, get_transcription, get_evaluation
from assr_tts.conversation import update_conversation, conversation_history
from workflow.claude import generate_response

def list_all_words():
    """
    Read all words from the data.json file.

    Returns:
        list: A list of word objects with their IDs and words.
    """
    try:
        with open("data/data.json", "r") as file:
            data = json.load(file)
            return data.get("words", [])
    except Exception as e:
        print(f"Error reading words data: {e}")
        return []

def get_word_by_id(word_id):
    """
    Get a specific word by its ID from the data.json file.

    Args:
        word_id (int): The ID of the word to retrieve.

    Returns:
        dict: The word data if found, None otherwise.
    """
    try:
        with open("data/data.json", "r") as file:
            data = json.load(file)
            words = data.get("words", [])

            for word in words:
                if word.get("id") == word_id:
                    return word

            return None
    except Exception as e:
        print(f"Error retrieving word with ID {word_id}: {e}")
        return None

# Define request model for Claude chat
class ClaudeRequest(BaseModel):
    message: str
    session_id: str

app = FastAPI()

@app.post("/transcribe")
async def get_transcript(audio_file: UploadFile = File(...)):
    audio_bytes = await audio_file.read()
    audio_path = f"temp_audio/{audio_file.filename}"
    os.makedirs("temp_audio", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    transcript = get_transcription(audio_path)
    print(f"Transcription ${transcript}")
    return transcript



@app.get("/get_speech/{text}")
def get_audio(text, language: str = "tw", speaker_id: str = "twi_speaker_8"):
    # Create directory for audio files if it doesn't exist
    os.makedirs("audio_output", exist_ok=True)

    # Generate a unique filename
    output_file = f"audio_output/{text.replace(' ', '_')}.wav"

    # Generate speech
    success, message = get_speech(text, language, speaker_id, output_file)

    if success:
        # Return the audio file
        return FileResponse(
            path=output_file,
            media_type="audio/wav",
            filename=Path(output_file).name
        )
    else:
        # Return error message
        raise HTTPException(status_code=500, detail=message)


@app.get("/words")
def get_all_words():
    """Get all available words with their IDs and meanings."""
    words = list_all_words()
    return JSONResponse(content={"words": words})


@app.get("/words/{word_id}")
def get_word(word_id: int):
    """Get a specific word by its ID.

    Args:
        word_id (int): The ID of the word to retrieve.

    Returns:
        JSONResponse: The word data if found, error message otherwise.
    """
    word_data = get_word_by_id(word_id)
    if word_data:
        return JSONResponse(content=word_data)
    else:
        raise HTTPException(status_code=404, detail=f"Word with ID {word_id} not found")


@app.post("/claude/chat")
async def claude_chat(request: ClaudeRequest):
    """
    Endpoint for persistent conversations with Claude.

    Args:
        request: Contains the user message and session ID

    Returns:
        A JSON response with Claude's reply and the conversation history
    """
    user_message = request.message
    session_id = request.session_id

    # Generate response from Claude using the session history
    claude_response = generate_response(user_message, session_id)

    # Get the updated conversation history
    history = conversation_history.get(session_id, [])

    # Find the index of the Claude response in the history
    # It should be the last system message
    message_index = None
    for i, msg in enumerate(history):
        if msg["role"] == "system" and msg["text"] == claude_response:
            message_index = i
            break

    # If we couldn't find the message in history, use the length of history
    if message_index is None and history:
        message_index = len(history) - 1

    # Generate audio for the response
    os.makedirs("claude_outputs", exist_ok=True)

    # Save with both message index and latest for backward compatibility
    if message_index is not None:
        audio_path = f"claude_outputs/{session_id}_{message_index}.wav"
        get_speech(claude_response, "tw", "twi_speaker_8", audio_path)

        # Also save as latest for backward compatibility
        latest_path = f"claude_outputs/{session_id}_latest.wav"
        get_speech(claude_response, "tw", "twi_speaker_8", latest_path)
    else:
        # Fallback to just latest if no message index
        audio_path = f"claude_outputs/{session_id}_latest.wav"
        get_speech(claude_response, "tw", "twi_speaker_8", audio_path)

    return {
        "message": user_message,
        "response": claude_response,
        "history": history,
        "audio_url": f"/claude/audio/{session_id}?message_index={message_index}" if message_index is not None else f"/claude/audio/{session_id}"
    }


@app.get("/claude/audio/{session_id}")
async def get_claude_audio(session_id: str, message_index: int = Query(None)):
    """
    Get the audio file for a specific Claude response for a session.

    Args:
        session_id: The session ID
        message_index: Optional index of the specific message to get audio for.
                      If not provided, returns the latest response.

    Returns:
        The audio file
    """
    if message_index is not None:
        audio_path = f"claude_outputs/{session_id}_{message_index}.wav"
    else:
        audio_path = f"claude_outputs/{session_id}_latest.wav"

    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav")
    else:
        return {"error": "Audio not found"}


@app.post("/claude/audio-chat")
async def claude_audio_chat(session_id: str = Query(...), audiofile: UploadFile = File(...)):
    """
    Endpoint for the complete workflow: audio input -> transcription -> Claude -> audio output.

    This endpoint takes an audio file, transcribes it, sends the transcription to Claude,
    and returns Claude's response as both text and audio, while maintaining conversation persistence.

    Args:
        session_id: The session ID for persistent conversation
        audiofile: The audio file containing the user's speech

    Returns:
        A JSON response with the transcription, Claude's reply, conversation history, and audio URL
    """
    # Save and transcribe the audio file
    audio_bytes = await audiofile.read()
    audio_path = f"temp_audio/{audiofile.filename}"
    os.makedirs("temp_audio", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # Transcribe the audio
    transcription = get_transcription(audio_path)
    os.remove(audio_path)

    # Generate response from Claude using the transcription and session history
    claude_response = generate_response(transcription, session_id)

    # Get the updated conversation history
    history = conversation_history.get(session_id, [])

    # Find the index of the Claude response in the history
    # It should be the last system message
    message_index = None
    for i, msg in enumerate(history):
        if msg["role"] == "system" and msg["text"] == claude_response:
            message_index = i
            break

    # If we couldn't find the message in history, use the length of history
    if message_index is None and history:
        message_index = len(history) - 1

    # Generate audio for the response
    os.makedirs("claude_outputs", exist_ok=True)

    # Save with both message index and latest for backward compatibility
    if message_index is not None:
        audio_path = f"claude_outputs/{session_id}_{message_index}.wav"
        get_speech(claude_response, "tw", "twi_speaker_8", audio_path)

        # Also save as latest for backward compatibility
        latest_path = f"claude_outputs/{session_id}_latest.wav"
        get_speech(claude_response, "tw", "twi_speaker_8", latest_path)
    else:
        # Fallback to just latest if no message index
        audio_path = f"claude_outputs/{session_id}_latest.wav"
        get_speech(claude_response, "tw", "twi_speaker_8", audio_path)

    return {
        "transcription": transcription,
        "response": claude_response,
        "history": history,
        "audio_url": f"/claude/audio/{session_id}?message_index={message_index}" if message_index is not None else f"/claude/audio/{session_id}"
    }


@app.post("/evaluation")
async def evaluate(id, audio_file: UploadFile = File(...)):
    audio_bytes = await audio_file.read()

    print(f"This is the ID: {id}")
    print(f"This is the audio file: {audio_file}")
    audio_path = f"temp_audio/{audio_file.filename}"
    os.makedirs("temp_audio", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    # print(id)
    return get_evaluation(audio_path, id)



# ct2-transformers-converter \
#     --model ./akan-non-standard-tiny \
#     --output_dir ./akan-non-standard-tiny/fast\
#     --quantization int8