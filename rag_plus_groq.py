from pathlib import Path
from demo_rag_offline import chunk_text, retrieve, answer as extractive_demo_answer
from llm_inference_groq import groq_answer

DATA_PATH = Path(__file__).parent / "DataSet" / "Raw_Text_Data.txt"

def build_context(question: str, top_k: int = 2) -> str:
    text = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    hits = retrieve(chunks, question, top_k=top_k)

    if not hits:
        return ""

    # Merge top chunks as LLM context
    return "\n\n".join([f"[CHUNK {i+1}]\n{chunk}" for i, (chunk, _) in enumerate(hits)])

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() or "What are the advantages of beekeeping?"

    # 1) Your offline extractive RAG (fallback demo)
    bullets, hits = extractive_demo_answer(q)
    print("\n=== OFFLINE RETRIEVAL (Preview / Extracted lines) ===\n")
    print(bullets)
    print("\nSOURCE CONFIDENCE SCORES:")
    for i, (_, score) in enumerate(hits, 1):
        print(f"  Chunk {i}: {score:.3f}")

    # 2) Groq grounded generation using retrieved chunks
    ctx = build_context(q, top_k=2)
    print("\n=== FINAL ANSWER (Groq LLM grounded in retrieved chunks) ===\n")
    if not ctx.strip():
        print("No context retrieved, cannot call Groq grounded.")
    else:
        print(groq_answer(q, context=ctx))