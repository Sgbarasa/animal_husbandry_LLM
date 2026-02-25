import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"

# Strongest option on Groq, if available on your account:
DEFAULT_MODEL = "llama-3.3-70b-versatile"
# Fallback if needed:
FALLBACK_MODEL = "llama-3.1-8b-instant"

def groq_answer(question: str, context: str, model: str = DEFAULT_MODEL) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    system = (
    "You are a friendly, practical livestock advisor speaking to a farmer.\n"
    "Use ONLY the provided context. Do not add new facts.\n"
    "If the context is insufficient, say: 'Not covered in the provided dataset.'\n"
    "Write in a natural conversational style: short paragraphs, simple words, clear steps.\n"
    "If you cite, put citations at the end of sentences like (CHUNK 1)."
)


    user = f"""Context:
{context}

Question:
{question}

Answer naturally like a helpful person, and add citations at the end of sentences:"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 350,
    }

    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    # If model name is not available, auto-fallback once
    if r.status_code != 200 and model == DEFAULT_MODEL:
        payload["model"] = FALLBACK_MODEL
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

    if r.status_code != 200:
        raise RuntimeError(f"Groq error {r.status_code}: {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"].strip()