import time, csv
from pathlib import Path
from llama_cpp import Llama

MODELS_DIR = Path(r"C:\Users\parig\PycharmProjects\quantized-edge-llm\models\llama-1b-mental-merged")
RESULTS_DIR = Path(r"C:\Users\parig\PycharmProjects\quantized-edge-llm\results")
RESULTS_DIR.mkdir(exist_ok=True)

GGUF_FILES = ["q4_0.gguf", "q4_k_m.gguf", "q5_k_m.gguf", "q8_0.gguf"]
PROMPT = "I have been feeling very anxious and overwhelmed lately. Can you help me understand what I might be going through?"

rows = []

for filename in GGUF_FILES:
    path = MODELS_DIR / filename
    if not path.exists():
        print(f"NOT FOUND: {path}")
        continue

    quant = filename.replace(".gguf", "").upper()
    size_gb = round(path.stat().st_size / 1e9, 2)
    print(f"\nBenchmarking {quant} ({size_gb} GB)...")

    try:
        llm = Llama(model_path=str(path), n_gpu_layers=-1, n_ctx=512, verbose=False)
        token_counts = []

        for run in range(3):
            start = time.perf_counter()
            output = llm(PROMPT, max_tokens=100, echo=False)
            elapsed = time.perf_counter() - start
            tokens = output["usage"]["completion_tokens"]
            tps = round(tokens / elapsed, 2)
            token_counts.append(tps)
            print(f"  Run {run+1}: {tps} tok/s")

        avg_tps = round(sum(token_counts) / 3, 2)
        print(f"  Average: {avg_tps} tok/s")
        rows.append({"device": "windows_rtx3050", "model": "llama-1b-mental", "quant": quant, "size_gb": size_gb, "avg_tps": avg_tps, "run1_tps": token_counts[0], "run2_tps": token_counts[1], "run3_tps": token_counts[2]})
        del llm

    except Exception as e:
        print(f"  ERROR: {e}")

if rows:
    csv_path = RESULTS_DIR / "windows_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {csv_path}")
    print("\nSUMMARY:")
    for r in rows:
        print(f"  {r['quant']:10} | {r['avg_tps']:>6} tok/s | {r['size_gb']} GB")
else:
    print("No results — check paths above")
