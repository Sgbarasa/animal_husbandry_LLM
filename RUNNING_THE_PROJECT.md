# Animal Husbandry LLM - Running the Project

> ⚠️ **IMPORTANT**: Gradient AI was acquired by DigitalOcean. You can still use it through DigitalOcean's platform!

---

## Virtual Environment (Recommended)

Yes, you should use a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Now install packages (they'll go into the venv, not system Python)
pip install torch transformers tqdm nltk spacy
```

---

## About Gradient AI (DigitalOcean)

You're right! Gradient AI was acquired by **DigitalOcean** (not Superagent). Let's check if you can still use it:

1. **Go to**: https://digitalocean.com/
2. **Sign up** for a DigitalOcean account
3. **Look for** "Gradient" or "AI Models" in their services
4. **Check** if they offer the same API endpoints

If Gradient is available through DigitalOcean:
- Get your new API credentials from DigitalOcean dashboard
- Update the API tokens in the code (they'll be different now)

---

## Option 1: Use Gradient AI (via DigitalOcean) - If Available

If DigitalOcean still offers Gradient AI:

```bash
pip install gradientai --upgrade
```

Then update the credentials in:
- `llm_inference.py` (lines 13-14)
- `train.py` (lines 12-13)
- `evaluate.py` (lines 15-16)

---

## Option 2: Use Hugging Face (Already in Project)

The Hugging Face version works independently and is already in the project.

---

## Step 1: Install Dependencies

```bash
# Core ML libraries
pip install torch transformers tqdm nltk spacy

# For RAG (Retrieval Augmented Generation)
pip install sentence-transformers joblib

# For training
pip install scikit-learn matplotlib pandas numpy evaluate

# For UI (optional)
pip install pyspellchecker pillow
```

---

## Step 2: Run the Hugging Face Code

The working code is in `Tests/test_Huggingface/`. Run these commands:

```bash
# Navigate to the folder
cd Tests/test_Huggingface

# Create DataStore folder for RAG embeddings
mkdir -p ../DataStore
```

### Option A: Run Training
```bash
python "LLM train.py"
```

### Option B: Run Inference (after training)
```bash
python "LLM test.py"
```

### Option C: Build RAG Embeddings
```bash
python RAG_retriver.py
```

---

## What Each File Does

| File | Purpose |
|------|---------|
| `LLM train.py` | Fine-tunes T5 model on animal husbandry data |
| `LLM test.py` | Interactive chat - ask questions |
| `RAG_retriver.py` | Builds embeddings from dataset for RAG |

---

## Hardware Requirements

- **Training**: Requires GPU (CUDA). Will run on CPU but very slow.
- **Inference**: Can run on CPU or GPU.
- **RAM**: At least 8GB recommended for the 13B model.

---

## If Training is Too Slow

The default model (`NousResearch/Nous-Hermes-Llama2-13b`) is large. To use a smaller model, edit `LLM train.py` line 24:

```python
# Change from:
model_name = "NousResearch/Nous-Hermes-Llama2-13b"

# To a smaller model:
model_name = "google/flan-t5-small"   # Very fast, CPU-friendly
# or
model_name = "google/flan-t5-base"     # Balance of speed and quality
```

---

## Data Location

- Training data: `DataSet/data.json`
- Raw text for RAG: `DataSet/Raw_Text_Data.txt`

The model saves to `qa_model/` and `qa_tokenizer/` folders after training.
