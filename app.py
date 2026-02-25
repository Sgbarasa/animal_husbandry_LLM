import streamlit as st

from demo_rag_offline import chunk_text, retrieve
from llm_inference_groq import groq_answer
from pathlib import Path

DATA_PATH = Path(__file__).parent / "DataSet" / "Raw_Text_Data.txt"

st.set_page_config(page_title="Animal Husbandry Assistant", page_icon="🐄", layout="centered")

st.title("🐄 Fugo AI (RAG + Groq)")
st.caption("Retrieves relevant info from your dataset, then generates a grounded answer with citations.")

# --- Helpers ---
def build_context(question: str, top_k: int = 2) -> str:
    text = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    hits = retrieve(chunks, question, top_k=top_k)

    if not hits:
        return "", []

    labeled_context = "\n\n".join([f"[CHUNK {i+1}]\n{chunk}" for i, (chunk, _) in enumerate(hits)])
    return labeled_context, hits


# --- UI ---
q = st.text_input("Ask a question", value="What are the advantages of beekeeping?")
top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=5, value=2)

col1, col2 = st.columns(2)
with col1:
    show_context = st.checkbox("Show retrieved context", value=True)
with col2:
    show_scores = st.checkbox("Show similarity scores", value=True)

if st.button("Ask", type="primary"):
    if not q.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Retrieving context..."):
            ctx, hits = build_context(q, top_k=top_k)

        if not ctx.strip():
            st.error("No relevant context retrieved from the dataset.")
        else:
            if show_context:
                st.subheader("Retrieved context (for grounding)")
                st.code(ctx[:2500] + ("..." if len(ctx) > 2500 else ""), language="text")

            if show_scores and hits:
                st.subheader("Chunk similarity scores")
                for i, (_, score) in enumerate(hits, 1):
                    st.write(f"CHUNK {i}: **{score:.3f}**")

            with st.spinner("Generating grounded answer..."):
                ans = groq_answer(q, context=ctx)

            st.subheader("Answer")
            st.write(ans)