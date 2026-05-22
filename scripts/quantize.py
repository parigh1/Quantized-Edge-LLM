# scripts/2_quantize.py
from llama_cpp import llama_model_quantize_params, llama_model_quantize, LLAMA_FTYPE_MOSTLY_Q4_0, LLAMA_FTYPE_MOSTLY_Q4_K_M, LLAMA_FTYPE_MOSTLY_Q5_K_M, LLAMA_FTYPE_MOSTLY_Q8_0
import ctypes

SRC = r"/models/llama-1b-mental-merged/model-f16.gguf"
OUT = r"C:\Users\parig\PycharmProjects\quantized-edge-llm\models\llama-1b-mental-merged"

quants = [
    ("q4_0",   LLAMA_FTYPE_MOSTLY_Q4_0),
    ("q4_k_m", LLAMA_FTYPE_MOSTLY_Q4_K_M),
    ("q5_k_m", LLAMA_FTYPE_MOSTLY_Q5_K_M),
    ("q8_0",   LLAMA_FTYPE_MOSTLY_Q8_0),
]

for name, ftype in quants:
    params = llama_model_quantize_params()
    params.nthread = 0
    params.ftype = ftype
    params.allow_requantize = False
    outfile = f"{OUT}\\{name}.gguf"
    print(f"Quantizing → {name}...")
    llama_model_quantize(
        SRC.encode(), outfile.encode(), ctypes.byref(params)
    )
    print(f"Done: {outfile}")