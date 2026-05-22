"""
3_benchmark.py — Benchmark inference across all quantized models.
Measures: load time, time-to-first-token, tokens/sec, peak RAM, file size.
Saves results to results/benchmark_results.csv
"""

import csv
import gc
import os
import time
import traceback
from pathlib import Path

import psutil

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(r"C:\Users\parig\PycharmProjects\quantized-edge-llm")
MDL_DIR  = BASE / "models" / "llama-1b-mental-merged"
RES_DIR  = BASE / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT  = RES_DIR / "benchmark_results.csv"

# ── Models to benchmark ───────────────────────────────────────────────────────
MODELS = [
    ("F16",    MDL_DIR / "model-f16.gguf"),
    ("Q8_0",   MDL_DIR / "q8_0.gguf"),
    ("Q5_K_M", MDL_DIR / "q5_k_m.gguf"),
    ("Q4_K_M", MDL_DIR / "q4_k_m.gguf"),
    ("Q4_0",   MDL_DIR / "q4_0.gguf"),
]

# ── Prompts ────────────────────────────────────────────────────────────────────
PROMPTS = [
    "I have been feeling very anxious and overwhelmed lately. What should I do?",
    "Can you explain the difference between depression and sadness?",
    "I can't sleep because of constant negative thoughts. How can I calm my mind?",
]

MAX_TOKENS   = 150
N_CTX        = 512
N_GPU_LAYERS = 0   # set to 35 if you want GPU offload

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_ram_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1e6

def benchmark_model(label, gguf_path):
    from llama_cpp import Llama

    if not gguf_path.exists():
        print(f"  [SKIP] {gguf_path.name} not found")
        return None

    file_mb = gguf_path.stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"  Model : {label}  ({file_mb:.0f} MB)")
    print(f"  File  : {gguf_path.name}")
    print(f"{'='*60}")

    ram_before = get_ram_mb()

    # ── Load ──────────────────────────────────────────────────────────────────
    t_load_start = time.perf_counter()
    try:
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )
    except Exception as e:
        print(f"  ✗ Load failed: {e}")
        return None

    load_time = time.perf_counter() - t_load_start
    ram_after_load = get_ram_mb()
    ram_used_mb = ram_after_load - ram_before
    print(f"  ✓ Loaded in {load_time:.2f}s  |  RAM delta: {ram_used_mb:.0f} MB")

    # ── Inference across prompts ───────────────────────────────────────────────
    total_tokens = 0
    total_time   = 0.0
    ttft_list    = []

    for i, prompt in enumerate(PROMPTS):
        full_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        t_start     = time.perf_counter()
        first_token = False
        ttft        = None
        tokens_out  = 0

        try:
            for chunk in llm(
                full_prompt,
                max_tokens=MAX_TOKENS,
                temperature=0.7,
                stream=True,
            ):
                if not first_token:
                    ttft = time.perf_counter() - t_start
                    first_token = True
                tokens_out += 1

        except Exception as e:
            print(f"  ✗ Inference error on prompt {i+1}: {e}")
            continue

        elapsed = time.perf_counter() - t_start
        tps     = tokens_out / elapsed if elapsed > 0 else 0

        total_tokens += tokens_out
        total_time   += elapsed
        if ttft:
            ttft_list.append(ttft)

        print(f"  Prompt {i+1}: {tokens_out} tokens | {tps:.1f} tok/s | TTFT {ttft*1000:.0f} ms")

    avg_tps  = total_tokens / total_time if total_time > 0 else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0

    print(f"\n  ► Avg: {avg_tps:.1f} tok/s | TTFT {avg_ttft*1000:.0f} ms | Load {load_time:.2f}s")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    del llm
    gc.collect()

    return {
        "label":        label,
        "file_mb":      round(file_mb, 1),
        "load_time_s":  round(load_time, 2),
        "ram_delta_mb": round(ram_used_mb, 1),
        "avg_tps":      round(avg_tps, 2),
        "avg_ttft_ms":  round(avg_ttft * 1000, 1),
        "total_tokens": total_tokens,
    }

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  quantized-edge-llm  |  Benchmark Suite")
    print("=" * 60)
    print(f"  Prompts : {len(PROMPTS)}")
    print(f"  Max tok : {MAX_TOKENS}")
    print(f"  GPU lyr : {N_GPU_LAYERS}")

    all_results = []

    for label, path in MODELS:
        try:
            result = benchmark_model(label, path)
            if result:
                all_results.append(result)
        except Exception:
            print(f"\n  ✗ Unexpected error for {label}:")
            traceback.print_exc()

    if not all_results:
        print("\nNo results to save.")
        return

    # ── Save CSV ───────────────────────────────────────────────────────────────
    fields = ["label", "file_mb", "load_time_s", "ram_delta_mb",
              "avg_tps", "avg_ttft_ms", "total_tokens"]

    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)

    print(f"\n\n{'='*60}")
    print(f"  Results saved → {CSV_OUT}")
    print(f"{'='*60}")

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\n{'Model':<10} {'MB':>6} {'Tok/s':>8} {'TTFT(ms)':>10} {'Load(s)':>9} {'RAM(MB)':>9}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['label']:<10} {r['file_mb']:>6.0f} {r['avg_tps']:>8.1f} "
              f"{r['avg_ttft_ms']:>10.0f} {r['load_time_s']:>9.2f} {r['ram_delta_mb']:>9.0f}")
    print("=" * 60)
    print("\nRun 4_plot_results.py to generate charts.\n")


if __name__ == "__main__":
    main()