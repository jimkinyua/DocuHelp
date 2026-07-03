from app.rag import generate_answer

result = generate_answer("what is this document about?",)
print("ANSWER:", result["answer"])
print("SOURCES:", result["sources"])