# scripts/convert_model.py
"""
Convert a Hugging Face sentence-transformer to OpenVINO IR.
• Works online or fully offline (local snapshot folder)
• Accepts an HF access token for private / rate-limited models
"""

import argparse, os, sys, json
from transformers import AutoTokenizer
from optimum.intel import OVModelForFeatureExtraction

# ─────────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="HF model ID *or* local snapshot folder")
parser.add_argument("--out",   default="../models/minilm_ov",
                    help="Target directory for IR files + tokenizer")
parser.add_argument("--token", default=os.getenv("HF_TOKEN"),
                    help="HF access token (env HF_TOKEN or --token)")
args = parser.parse_args()

MODEL_ID       = args.model
OUTPUT_DIR     = args.out
HF_AUTH_TOKEN  = args.token

print(f"── Converting: {MODEL_ID} → {OUTPUT_DIR}")
if HF_AUTH_TOKEN:
    print("• Using Hugging Face token for download/auth")

# ─────────────────────────── Prep dirs ──────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    # Step 1 – Tokenizer
    print("Step 1: Download / copy tokenizer …")
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_AUTH_TOKEN,            # None if offline/local
        trust_remote_code=True)
    tok.save_pretrained(OUTPUT_DIR)
    print("  ✓ Tokenizer saved")

    # Step 2 – Model → IR
    print("Step 2: Exporting model to OpenVINO IR …")
    ov_model = OVModelForFeatureExtraction.from_pretrained(
        MODEL_ID,
        export=True,                    # triggers ONNX→IR under the hood
        token=HF_AUTH_TOKEN)
    ov_model.save_pretrained(OUTPUT_DIR)
    print("  ✓ Model IR saved")

    # Small manifest so you remember what was exported
    json.dump({"source": MODEL_ID}, open(os.path.join(OUTPUT_DIR, "manifest.json"), "w"))
    print("\n🎉  Conversion complete")

except Exception as exc:
    print("\n🛑  Conversion failed")
    print(f"Reason: {exc}")
    print("• Check model ID spelling or network\n• If offline, "
          "download with `huggingface-cli download` then pass the local path")
    sys.exit(1)
