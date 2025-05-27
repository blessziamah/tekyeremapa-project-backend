# In-memory conversation history
conversation_history = {}  # key: session_id, value: list of {"role": str, "text": str}


def update_conversation(session_id: str, role: str, text: str):
    """
    Update the conversation history for a session.
    
    Args:
        session_id (str): The session ID
        role (str): The role of the message sender ("user" or "system")
        text (str): The message text
    """
    history = conversation_history.setdefault(session_id, [])
    history.append({"role": role, "text": text})
    if len(history) > 10:
        history.pop(0)


def generate_nlp_response(session_id: str) -> str:
    """
    Simple NLP rule-based response generator for kids.
    You can later swap this with an actual LLM or finetuned model.
    
    Args:
        session_id (str): The session ID
        
    Returns:
        str: A response based on the last user input
    """
    history = conversation_history.get(session_id, [])
    if not history:
        return "Hello! Let's begin."

    last_user_input = history[-1]["text"].lower()

    if "hello" in last_user_input or "hi" in last_user_input:
        return "Hi there! Can you repeat after me?"

    elif "done" in last_user_input or "finished" in last_user_input:
        return "Good job! Want to try something new?"

    elif "no" in last_user_input or "not" in last_user_input:
        return "That's okay! Take your time."

    elif "again" in last_user_input or "retry" in last_user_input:
        return "Sure! Let's try again."

    else:
        return "Great! Keep going!"