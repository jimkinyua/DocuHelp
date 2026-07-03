import traceback
from app.rag import generate_answer

messages = [{"role": "user", "content": "what is this document about?"}]
try:
    print("=== Testing Google ===")
    result_google = generate_answer(messages, provider="google")
    print("ANSWER:", result_google["answer"][:100])
    
    print("\n=== Testing Groq ===")
    result_groq = generate_answer(messages, provider="groq")
    print("ANSWER:", result_groq["answer"][:100])
except Exception as e:
    traceback.print_exc()