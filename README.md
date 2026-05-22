# Quantized Edge LLM: From HuggingFace to Raspberry Pi 4

Benchmarking quantized LLM inference across NVIDIA GPU and Raspberry Pi 4 ARM CPU,
using my own fine-tuned Llama 3.2-1B model trained for mental health conversations.

## Models Tested

| Model | HuggingFace | Type |
|---|---|---|
| Llama 3.2-1B Mental Health | [Parigh1/Llama-3.2-1B-Mental-Health-Friend](https://huggingface.co/Parigh1/Llama-3.2-1B-Mental-Health-Friend) | LoRA fine-tune merged |
| Gugu 8B (Mental Wellness) | [Parigh1/gugu-merged](https://huggingface.co/Parigh1/gugu-merged) | Full fine-tune |

## Windows GPU Results — RTX 3050 Laptop (4GB VRAM)

| Quantization | Size | Speed (tok/s) | Notes |
|---|---|---|---|
| Q4_K_M | 1.64 GB | **91.54** | Fastest — mixed precision is GPU-friendly |
| Q4_0 | 1.61 GB | 88.37 | Slightly slower despite lower precision |
| Q5_K_M | 1.75 GB | 77.61 | Speed drop not proportional to size increase |
| Q8_0 | 2.09 GB | 76.39 | Slowest — VRAM pressure limits throughput |

**Key finding:** Q4_K_M outperforms Q4_0 in speed on GPU despite higher quality, due to better memory access patterns in K-quant variants.

## Raspberry Pi 4 Results (ARM CPU, 4GB RAM)

*(Coming soon — benchmarks in progress)*

## Pipeline

1. Download LoRA adapter from HuggingFace (scripts/1_download_models.py)
2. Merge adapter with base model using PEFT (scripts/merge_adapter.py)
3. Convert merged model to GGUF format (llama.cpp convert_hf_to_gguf.py)
4. Quantize to 4 levels using llama-cpp-python (scripts/quantize_script.py)
5. Benchmark on Windows GPU (scripts/benchmark_gpu.py)
6. Benchmark on Raspberry Pi 4 (pi4/benchmark_rpi.py) — in progress

## Hardware

- **Development + GPU benchmark**: Windows 11, NVIDIA RTX 3050 Laptop (4GB VRAM)
- **Edge target**: Raspberry Pi 4 (4GB RAM, ARM Cortex-A72, CPU-only)

## Key Findings

- Q4_K_M is the optimal quantization for GPU inference — fastest despite higher quality than Q4_0
- Q8_0 bottlenecks on VRAM bandwidth at 2.09GB, reducing throughput below smaller quantizations
- All 4 quantization levels run comfortably within 4GB VRAM on the 1B model
- 8B model (Gugu) at Q4_K_M (~4.7GB) exceeds RPi4's 4GB RAM — sub-Q4 required for edge deployment

## Setup

`ash
git clone https://github.com/parigh1/Quantized-Edge-LLM
cd Quantized-Edge-LLM
pip install -r requirements.txt
`

---
*Part of my edge AI + LLM portfolio — also see [EVA-1 Drone CV System](https://github.com/parigh1)*
