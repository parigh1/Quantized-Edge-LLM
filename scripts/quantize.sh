LLAMA="C:/llama.cpp/llama-quantize.exe"
MODELS_DIR="./models"

for MODEL in "llama-1b-mental" "gugu-merged"; do
    SRC="$MODELS_DIR/$MODEL/model-f16.gguf"
    OUT="$MODELS_DIR/$MODEL"

    echo "Quantizing $MODEL..."
    "$LLAMA" "$SRC" "$OUT/q4_0.gguf"   Q4_0
    "$LLAMA" "$SRC" "$OUT/q4_k_m.gguf" Q4_K_M
    "$LLAMA" "$SRC" "$OUT/q5_k_m.gguf" Q5_K_M
    "$LLAMA" "$SRC" "$OUT/q8_0.gguf"   Q8_0
    echo "Done: $MODEL"
done