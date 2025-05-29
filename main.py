import os
import json
import base64
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body, Path as FastAPIPath
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from assr_tts.get_models import get_speech, get_transcription, get_evaluation
from assr_tts.conversation import update_conversation, conversation_history
from workflow.claude import generate_response

# Dictionary to store hardcoded conversation states
hardcoded_conversation_states = {}

# Dictionary to store full conversation flow states
full_conversation_states = {}

def load_hardcoded_conversations():
    """
    Load the hardcoded conversations from convo_data.json.

    Returns:
        list: A list of conversation pairs with prompts and responses.
    """
    try:
        with open("data/convo_data.json", "r") as file:
            data = json.load(file)
            return data.get("conversations", [])
    except Exception as e:
        print(f"Error reading conversation data: {e}")
        return []

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



@app.post("/hardcoded-conversation/start")
async def start_hardcoded_conversation(session_id: str = Query(...)):
    """
    Start a new hardcoded conversation flow.

    Args:
        session_id (str): The session ID for tracking the conversation state

    Returns:
        The audio file for the first prompt directly
    """
    # Load the hardcoded conversations
    conversations = load_hardcoded_conversations()

    if not conversations:
        raise HTTPException(status_code=500, detail="No conversation data available")

    # Reset the conversation state for this session
    hardcoded_conversation_states[session_id] = {
        "current_index": 0,
        "total_pairs": len(conversations),
        "completed": False
    }

    # Get the first prompt
    first_prompt = conversations[0]["prompt"]

    # Generate audio for the first prompt
    os.makedirs("audio_output", exist_ok=True)
    audio_path = f"audio_output/{session_id}_prompt_0.wav"
    get_speech(first_prompt, "tw", "twi_speaker_8", audio_path)

    # Return the audio file directly
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav")
    else:
        raise HTTPException(status_code=500, detail="Failed to generate audio")

@app.get("/hardcoded-conversation/status/{session_id}")
async def get_hardcoded_conversation_status(session_id: str):
    """
    Get the current status of a hardcoded conversation.

    Args:
        session_id (str): The session ID

    Returns:
        A JSON response with the current conversation state
    """
    # Check if the session exists
    if session_id not in hardcoded_conversation_states:
        raise HTTPException(status_code=404, detail="Conversation session not found")

    # Get the current state
    state = hardcoded_conversation_states[session_id]

    # Load the hardcoded conversations
    conversations = load_hardcoded_conversations()

    # Get the current prompt
    current_index = state["current_index"]
    current_prompt = conversations[current_index]["prompt"] if current_index < len(conversations) else None

    return {
        "session_id": session_id,
        "current_index": current_index,
        "total_prompts": state["total_pairs"],
        "current_prompt": current_prompt,
        "completed": state["completed"],
        "audio_url": f"/hardcoded-conversation/audio/{session_id}?prompt_index={current_index}" if current_prompt else None
    }

@app.get("/hardcoded-conversation/audio/{session_id}")
async def get_hardcoded_conversation_audio(session_id: str, prompt_index: int = Query(...)):
    """
    Get the audio file for a specific prompt in the hardcoded conversation.

    Args:
        session_id (str): The session ID
        prompt_index (int): The index of the prompt to get audio for

    Returns:
        The audio file
    """
    audio_path = f"audio_output/{session_id}_prompt_{prompt_index}.wav"

    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav")
    else:
        raise HTTPException(status_code=404, detail="Audio not found")

@app.post("/hardcoded-conversation/respond")
async def respond_to_hardcoded_conversation(
    session_id: str = Query(...),
    audiofile: UploadFile = File(...)
):
    """
    Process a user's audio response in the hardcoded conversation flow.

    Args:
        session_id (str): The session ID for tracking the conversation state
        audiofile: The audio file containing the user's response

    Returns:
        A JSON response with the next prompt or completion message
    """
    # Check if the session exists
    if session_id not in hardcoded_conversation_states:
        raise HTTPException(status_code=404, detail="Conversation session not found")

    # Get the current state
    state = hardcoded_conversation_states[session_id]

    # Check if the conversation is already completed
    if state["completed"]:
        raise HTTPException(status_code=400, detail="Conversation already completed")

    # Load the hardcoded conversations
    conversations = load_hardcoded_conversations()

    # Save and transcribe the audio file
    audio_bytes = await audiofile.read()
    audio_path = f"temp_audio/{audiofile.filename}"
    os.makedirs("temp_audio", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # Transcribe the audio
    transcription = get_transcription(audio_path)
    os.remove(audio_path)

    # Get the expected response for the current prompt
    current_index = state["current_index"]
    expected_response = conversations[current_index]["response"]

    # Move to the next prompt
    next_index = current_index + 1

    # Check if we've reached the end of the conversation
    if next_index >= state["total_pairs"]:
        hardcoded_conversation_states[session_id]["completed"] = True
        return {
            "session_id": session_id,
            "transcription": transcription,
            "expected_response": expected_response,
            "completed": True,
            "message": "Conversation completed"
        }

    # Update the conversation state
    hardcoded_conversation_states[session_id]["current_index"] = next_index

    # Get the next prompt
    next_prompt = conversations[next_index]["prompt"]

    # Generate audio for the next prompt
    audio_path = f"audio_output/{session_id}_prompt_{next_index}.wav"
    get_speech(next_prompt, "tw", "twi_speaker_8", audio_path)

    return {
        "session_id": session_id,
        "transcription": transcription,
        "expected_response": expected_response,
        "prompt": next_prompt,
        "prompt_index": next_index,
        "total_prompts": state["total_pairs"],
        "audio_url": f"/hardcoded-conversation/audio/{session_id}?prompt_index={next_index}",
        "completed": False
    }

# ct2-transformers-converter \
#     --model ./akan-non-standard-tiny \
#     --output_dir ./akan-non-standard-tiny/fast\
#     --quantization int8

@app.post("/full-conversation-flow-audio")
async def full_conversation_flow_audio(session_id: str = Query(...), audio_file: UploadFile = File(None)):
    """
    Handle the entire conversation flow and return the audio file directly.

    This endpoint works similarly to /full-conversation-flow but returns the audio WAV file
    directly instead of a JSON response with base64-encoded audio data.

    Args:
        session_id (str): The session ID for tracking the conversation state
        audio_file (UploadFile, optional): The audio file containing the user's response.
                                          Not required for the first request.

    Returns:
        A WAV audio file response
    """
    # Load the hardcoded conversations
    conversations = load_hardcoded_conversations()

    if not conversations:
        raise HTTPException(status_code=500, detail="No conversation data available")

    # Check if this is a new session or continuing an existing one
    if session_id not in full_conversation_states:
        # Initialize a new conversation
        full_conversation_states[session_id] = {
            "current_index": 0,
            "total_pairs": len(conversations),
            "completed": False,
            "history": []
        }

        # Get the first prompt
        current_prompt = conversations[0]["prompt"]

        # Generate audio for the first prompt
        os.makedirs("audio_output", exist_ok=True)
        audio_path = f"audio_output/{session_id}_prompt_0.wav"
        get_speech(current_prompt, "tw", "twi_speaker_8", audio_path)

        # Return the audio file directly
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"{session_id}_prompt_0.wav"
        )

    # Get the current state
    state = full_conversation_states[session_id]

    # Check if the conversation is already completed
    if state["completed"]:
        raise HTTPException(status_code=400, detail="Conversation already completed")

    # Process the user's audio response
    if audio_file:
        # Save and transcribe the audio file
        audio_bytes = await audio_file.read()
        audio_path = f"temp_audio/{audio_file.filename}"
        os.makedirs("temp_audio", exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Transcribe the audio
        transcription = get_transcription(audio_path)
        os.remove(audio_path)

        # Get the expected response for the current prompt
        current_index = state["current_index"]
        expected_response = conversations[current_index]["response"]

        # Add to history
        state["history"].append({
            "prompt": conversations[current_index]["prompt"],
            "expected_response": expected_response,
            "user_response": transcription
        })

        # Move to the next prompt
        next_index = current_index + 1

        # Check if we've reached the end of the conversation
        if next_index >= state["total_pairs"]:
            full_conversation_states[session_id]["completed"] = True
            raise HTTPException(status_code=400, detail="Conversation completed")

        # Update the conversation state
        full_conversation_states[session_id]["current_index"] = next_index

        # Get the next prompt
        next_prompt = conversations[next_index]["prompt"]

        # Generate audio for the next prompt
        audio_path = f"audio_output/{session_id}_prompt_{next_index}.wav"
        get_speech(next_prompt, "tw", "twi_speaker_8", audio_path)

        # Return the audio file directly
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"{session_id}_prompt_{next_index}.wav"
        )
    else:
        # If no audio file is provided but it's not a new session,
        # return the current prompt's audio again
        current_index = state["current_index"]
        audio_path = f"audio_output/{session_id}_prompt_{current_index}.wav"

        # Return the audio file directly
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"{session_id}_prompt_{current_index}.wav"
        )

@app.post("/full-conversation-flow")
async def full_conversation_flow(session_id: str = Query(...), audio_file: UploadFile = File(None)):
    """
    Handle the entire conversation flow in a single endpoint.

    This endpoint manages the full conversation flow using the hardcoded data from convo_data.json.
    It starts with the avatar reading the first prompt, then processes user audio responses
    and continues the conversation until completion.

    Args:
        session_id (str): The session ID for tracking the conversation state
        audio_file (UploadFile, optional): The audio file containing the user's response.
                                          Not required for the first request.

    Returns:
        A JSON response with the current prompt, audio data (base64 encoded), and conversation state
    """
    # Load the hardcoded conversations
    conversations = load_hardcoded_conversations()

    if not conversations:
        raise HTTPException(status_code=500, detail="No conversation data available")

    # Check if this is a new session or continuing an existing one
    if session_id not in full_conversation_states:
        # Initialize a new conversation
        full_conversation_states[session_id] = {
            "current_index": 0,
            "total_pairs": len(conversations),
            "completed": False,
            "history": []
        }

        # Get the first prompt
        current_prompt = conversations[0]["prompt"]

        # Generate audio for the first prompt
        os.makedirs("audio_output", exist_ok=True)
        audio_path = f"audio_output/{session_id}_prompt_0.wav"
        get_speech(current_prompt, "tw", "twi_speaker_8", audio_path)

        # Read the audio file and encode it as base64
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return {
            "session_id": session_id,
            "prompt": current_prompt,
            "prompt_index": 0,
            "total_prompts": len(conversations),
            "audio_data": audio_base64,
            "audio_format": "wav",
            "completed": False,
            "history": []
        }

    # Get the current state
    state = full_conversation_states[session_id]

    # Check if the conversation is already completed
    if state["completed"]:
        return {
            "session_id": session_id,
            "completed": True,
            "message": "Conversation already completed",
            "history": state["history"]
        }

    # Process the user's audio response
    if audio_file:
        # Save and transcribe the audio file
        audio_bytes = await audio_file.read()
        audio_path = f"temp_audio/{audio_file.filename}"
        os.makedirs("temp_audio", exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Transcribe the audio
        transcription = get_transcription(audio_path)
        os.remove(audio_path)

        # Get the expected response for the current prompt
        current_index = state["current_index"]
        expected_response = conversations[current_index]["response"]

        # Add to history
        state["history"].append({
            "prompt": conversations[current_index]["prompt"],
            "expected_response": expected_response,
            "user_response": transcription
        })

        # Move to the next prompt
        next_index = current_index + 1

        # Check if we've reached the end of the conversation
        if next_index >= state["total_pairs"]:
            full_conversation_states[session_id]["completed"] = True
            return {
                "session_id": session_id,
                "transcription": transcription,
                "expected_response": expected_response,
                "completed": True,
                "message": "Conversation completed",
                "history": state["history"]
            }

        # Update the conversation state
        full_conversation_states[session_id]["current_index"] = next_index

        # Get the next prompt
        next_prompt = conversations[next_index]["prompt"]

        # Generate audio for the next prompt
        audio_path = f"audio_output/{session_id}_prompt_{next_index}.wav"
        get_speech(next_prompt, "tw", "twi_speaker_8", audio_path)

        # Read the audio file and encode it as base64
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return {
            "session_id": session_id,
            "transcription": transcription,
            "expected_response": expected_response,
            "prompt": next_prompt,
            "prompt_index": next_index,
            "total_prompts": state["total_pairs"],
            "audio_data": audio_base64,
            "audio_format": "wav",
            "completed": False,
            "history": state["history"]
        }
    else:
        # If no audio file is provided but it's not a new session,
        # return the current prompt again
        current_index = state["current_index"]
        current_prompt = conversations[current_index]["prompt"]

        # Read the audio file and encode it as base64
        audio_path = f"audio_output/{session_id}_prompt_{current_index}.wav"
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return {
            "session_id": session_id,
            "prompt": current_prompt,
            "prompt_index": current_index,
            "total_prompts": state["total_pairs"],
            "audio_data": audio_base64,
            "audio_format": "wav",
            "completed": False,
            "history": state["history"]
        }
