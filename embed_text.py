import numpy as np
import faiss
from openvino.runtime import Core
from transformers import AutoTokenizer
from pathlib import Path

# --- Load tokenizer & model ---
MODEL_DIR = Path("../models/minilm_ov")
core = Core()
compiled = core.compile_model(MODEL_DIR / "openvino_model.xml", "CPU")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# --- Embed input text ---
def embed_text(text):
    # Tokenize and convert to plain dict of NumPy arrays
    enc = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    inputs = {k: v for k, v in enc.items()}  # <- convert from BatchEncoding

    # Pass the full dictionary of expected inputs
    outputs = compiled(inputs)
    
    # Get the vector output
    vector = outputs[compiled.output(0)]
    
    # Optional: Mean pooling over tokens
    return np.mean(vector, axis=1)



# --- Get the actual vector ---
text = "What is the shape of longing when memory folds?"
vector = embed_text(text)

# --- FAISS Setup ---
d = 384
faiss.omp_set_num_threads(8)
quant = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quant, d, 64, 8, 8)

# --- Train & Add ---
train_vecs = np.random.randn(40000, d).astype("float32")
index.train(train_vecs)
index.add(train_vecs)

# --- Search ---
index.nprobe = 48
D, I = index.search(vector.astype("float32"), k=5)
print("Nearest IDs:", I)
