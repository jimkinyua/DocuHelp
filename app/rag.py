import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
from google import genai
from google.genai import types
from app.vectorstore import search_chunks

load_dotenv()

google_key = os.environ.get("GOOGLE_API_KEY")
google_client = genai.Client(api_key=google_key) if google_key else None

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", "missing_key"),
)

MODEL = "llama-3.3-70b-versatile"


def build_prompt(question: str, results) -> str:
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = ""
    for doc, meta in zip(docs, metas):
        source = meta["source"]
        idx = meta["chunk_index"]
        context += f"[Source: {source}, chunk {idx}]\n{doc}\n\n"

    return (
        "You are a careful assistant. Answer the question using ONLY the context "
        "below. If the answer is not in the context, say you don't have that "
        "information. Always cite the source and chunk you used.\n\n"
        f"CONTEXT:\n{context}\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


def generate_answer(messages: list[dict], source: str = None, top_k: int = 4, provider: str = "groq") -> dict:
    question = messages[-1]["content"] if messages else ""
    results = search_chunks(question, source=source, top_k=top_k)
    prompt = build_prompt(question, results)

    llm_messages = messages[:-1] + [{"role": "user", "content": prompt}]

    try:
        if provider == "google":
            if not google_client:
                answer = "Google API key is not configured on this server."
            else:
                gemini_messages = []
                for msg in llm_messages:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
                
                response = google_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=gemini_messages
                )
                answer = response.text
        else:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=llm_messages,
            )
            answer = completion.choices[0].message.content
    except openai.RateLimitError:
        answer = "We are currently on a free tier and have reached our rate limit. Please try again in a few minutes."
    except Exception as e:
        answer = f"An error occurred while generating the answer: {str(e)}"

    sources = [
        {"source": m["source"], "chunk_index": m["chunk_index"]}
        for m in results["metadatas"][0]
    ]
    return {"answer": answer, "sources": sources}