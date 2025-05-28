import anthropic
from dotenv import load_dotenv
import os

from assr_tts.conversation import update_conversation, conversation_history
from assr_tts.get_models import get_speech

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_KEY")
)

system_prompt = """
You are an assistant for speech-impaired children who speak Twi. When responding:
- Use simple Twi vocabulary appropriate for children aged 4-8
- Keep sentences very short (3-5 words maximum)
- Speak slowly
- Be extremely patient and encouraging
- Recognize that pronunciation may be imperfect
- When appropriate, offer simple word choices for the child to select
- Always maintain a warm, supportive tone
- Never return english in your response
- Remove new line characters in your response
"""


def generate_response(user_input, session_id=None):
    try:
        # If session_id is provided, use conversation history
        if session_id:
            # Add the current user input to conversation history
            update_conversation(session_id, "user", user_input)

            # Get conversation history for this session
            history = conversation_history.get(session_id, [])

            # Convert history to the format expected by Claude API
            messages = []
            for msg in history:
                messages.append({"role": "user" if msg["role"] == "user" else "assistant",
                                 "content": msg["text"]})

            # If this is a new conversation, just use the current message
            if not messages:
                messages = [{"role": "user", "content": user_input}]
        else:
            # No session_id provided, just use the current message (no persistence)
            messages = [{"role": "user", "content": user_input}]

        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=system_prompt,
            max_tokens=150,  # Keep responses brief
            temperature=0.3,  # More consistent/predictable responses
            messages=messages
        )

        response_text = message.content[0].text

        # If session_id is provided, add the response to conversation history
        if session_id:
            update_conversation(session_id, "system", response_text)

        return response_text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Me te wo. Me boa wo."

# # Example usage (for testing)
# if __name__ == "__main__":
#     result = generate_response("Wo ho te sen?")
#     print(result)
#     get_speech(result, "tw", "twi_speaker_8", "output_claude.wav")
#
#     # Test with session persistence
#     test_session = "test_session_123"
#     print("\nTesting with session persistence:")
#
#     # First message
#     response1 = generate_response("Wo ho te sen?", test_session)
#     print("User: Wo ho te sen?")
#     print(f"Claude: {response1}")
#
#     # Second message in same session
#     response2 = generate_response("Me din de Kofi", test_session)
#     print("User: Me din de Kofi")
#     print(f"Claude: {response2}")
