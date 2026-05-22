import subprocess, csv, time, os, psutil, json
from pathlib import Path

LLAMA_BENCH = "C:/llama.cpp/llama-bench.exe"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CONFIGS = [
    ("llama-1b-mental", "q4_0"),
    ("llama-1b-mental", "q4_k_m"),
    ("llama-1b-mental", "q5_k_m"),
    ("llama-1b-mental", "q8_0"),
    ("gugu-merged", "q4_0"),
    ("gugu-merged", "q4_k_m"),
    ("gugu-merged", "q5_k_m"),
    # skip q8_0 for 8B unless you have >16GB VRAM
]

rows = []
for model_name, quant in CONFIGS:
    gguf = MODELS_DIR / model_name / f"{quant}.gguf"
    if not gguf.exists():
        print(f"Skipping {model_name}/{quant} — file not found")
        continue

    print(f"\nBenchmarking {model_name} {quant}...")

    # llama-bench outputs JSON
    cmd = [
        LLAMA_BENCH,
        "-m", str(gguf),
        "-ngl", "99",  # offload all layers to GPU
        "-p", "512",  # prompt tokens
        "-n", "128",  # tokens to generate
        "-r", "3",  # repeat 3 times for average
        "--output", "json",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        for entry in data:
            rows.append({
                "device": "windows_gpu",
                "model": model_name,
                "quant": quant.upper(),
                "model_size_gb": round(gguf.stat().st_size / 1e9, 2),
                "pp_tps": round(entry.get("pp_tokens_per_second", 0), 1),
                "tg_tps": round(entry.get("tg_tokens_per_second", 0), 1),
            })
            print(f"  Prompt: {rows[-1]['pp_tps']} t/s | Generate: {rows[-1]['tg_tps']} t/s")
    except Exception as e:
        print(f"  Parse error: {e}\n  Raw: {result.stdout[:200]}")

csv_path = RESULTS_DIR / "windows_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved to {csv_path}")