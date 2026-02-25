import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "DataSet" / "Raw_Text_Data.txt"

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def retrieve(chunks, query, top_k=3):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )
    X = vectorizer.fit_transform(chunks)
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).ravel()

    ranked_indices = sims.argsort()[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in ranked_indices if sims[i] > 0.05]

def extractive_answer(context: str, query: str, max_sentences: int = 4):
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]

    # Clean + filter out broken sentences and duplicates
    seen = set()
    sentences = []

    bad_starts = ("but ", "and ", "or ", "so ", "then ")
    bad_phrases = ("but all of th",)

    for s in raw_sentences:
        s = re.sub(r"\s+", " ", s).strip()
        s_lower = s.lower()

        # remove obvious broken lines
        if any(p in s_lower for p in bad_phrases):
            continue
        if s_lower.startswith(bad_starts):
            continue
        if len(s) < 45:
            continue

        # strong de-dupe
        key = re.sub(r"[^a-z0-9]+", " ", s_lower).strip()
        if key in seen:
            continue
        seen.add(key)

        sentences.append(s)

    if not sentences:
        return "No clean answer sentences found in the dataset."

    # Rank sentences by similarity to query
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences)
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).ravel()

    ranked_idx = sims.argsort()[::-1]

    chosen = []
    for i in ranked_idx:
        if sims[i] < 0.06:   # keep it lenient so we don't lose list sentences
            continue
        chosen.append(sentences[i])
        if len(chosen) >= max_sentences:
            break

    if not chosen:
        chosen = [sentences[ranked_idx[0]]]

    # Clean awkward prefixes from dataset text
    cleaned_out = []
    for s in chosen:
        s = re.sub(r"^apiculture\s*what\s*is\s*beekeeping\s*:\s*", "", s, flags=re.I)
        s = re.sub(r"^what\s*is\s*beekeeping\s*:\s*", "", s, flags=re.I)
        cleaned_out.append(s)

    # Remove near-duplicate sentences (high token overlap)
    final = []
    for s in cleaned_out:
        s_tokens = set(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())
        is_dup = False
        for t in final:
            t_tokens = set(re.sub(r"[^a-z0-9]+", " ", t.lower()).split())
            overlap = len(s_tokens & t_tokens) / max(1, min(len(s_tokens), len(t_tokens)))
            if overlap > 0.75:
                is_dup = True
                break
        if not is_dup:
            final.append(s)

    return "\n".join(f"- {s}" for s in final)

def answer(query: str, top_k: int = 3):
    text = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    hits = retrieve(chunks, query, top_k=top_k)

    if not hits:
        return "No relevant content found.", []

    # Use top 2 chunks to avoid missing key list sentences across chunk boundaries
    merged_context = "\n\n".join([chunk for chunk, _ in hits[:2]])

    ans = extractive_answer(merged_context, query)

    return ans, hits

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() or "What is animal husbandry and why is it important?"
    ans, hits = answer(q)

    print("\nANSWER (Grounded Retrieval Response):\n")
    print(ans)

    print("\nSOURCE CONFIDENCE SCORES:")
    for i, (_, score) in enumerate(hits, 1):
        print(f"  Chunk {i}: {score:.3f}")