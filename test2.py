from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",  # Use Mixtral if supported
    output="E:/data/onnx/mistral7b",
    task="text-generation"
)
