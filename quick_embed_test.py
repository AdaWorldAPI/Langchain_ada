# quick_embed_test.py  (run from your project root)
from openvino import Core
from transformers import AutoTokenizer
import numpy as np, time, pathlib

MODEL_DIR = pathlib.Path(r"E:\data\models\minilm_ov")   # <─ your folder

core      = Core()
compiled  = core.compile_model(MODEL_DIR / "openvino_model.xml", "CPU")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

text   = ["The Soulframe engine feels the ripple behind the words."]
toks   = tokenizer(text, return_tensors="np", padding=True)

# OpenVINO IR expects 3 inputs (see XML): input_ids, attention_mask, token_type_ids
if "token_type_ids" not in toks:
    toks["token_type_ids"] = np.zeros_like(toks["input_ids"])

feed_dict = {
    "input_ids":      toks["input_ids"],
    "attention_mask": toks["attention_mask"],
    "token_type_ids": toks["token_type_ids"],
}

start   = time.time()
output  = compiled(feed_dict)
elapsed = (time.time() - start) * 1000

# first (and only) output
vector  = output[compiled.output(0)]
print("vector shape:", vector.shape, "| latency:", round(elapsed, 2), "ms")
print("sample slice:", vector[0, :8])  # show a few numbers
