from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
dummy_input = torch.randint(0, 10000, (1, 128))
torch.onnx.export(model, (dummy_input,), "minilm.onnx", input_names=["input_ids"], output_names=["last_hidden_state"])
print("ONNX model exported successfully.")