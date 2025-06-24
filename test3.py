from llama_cpp import Llama

llm = Llama(model_path="mixtral.Q4_K_M.gguf", n_ctx=4096, n_threads=8, n_gpu_layers=0)
response = llm("### Human: Write a haiku about thunder\n### Assistant:", max_tokens=128)
print(response["choices"][0]["text"])
