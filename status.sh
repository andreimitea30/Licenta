#!/usr/bin/env bash
ROOT="/c/Users/mitea/Desktop/Licenta"
EXTRACT_DIR="$ROOT/extracted_skeletons_hierarchical"
EXTRACT_TASK_LOG="/c/Users/mitea/AppData/Local/Temp/claude/C--Users-mitea-Desktop-IDP/82293a43-ca05-4b93-b908-31e088394be5/tasks/bpju6jo2z.output"
TRAIN_TASK_LOG="/c/Users/mitea/AppData/Local/Temp/claude/C--Users-mitea-Desktop-IDP/82293a43-ca05-4b93-b908-31e088394be5/tasks/bzh5ld1vi.output"
TRAIN_LOG="$ROOT/compare_v6_combined.log"
TASK_LOG="$EXTRACT_TASK_LOG"

echo "=== Now ==="
date
echo

echo "=== Extraction progress ==="
n=$(find "$EXTRACT_DIR" -name "*.npy" 2>/dev/null | wc -l)
echo "Files extracted: $n / 10000"
echo
echo "Latest tqdm per split:"
for split in train val test; do
    tail -300 "$TASK_LOG" 2>/dev/null | grep -oE "${split} \(x8\):.*\| [0-9]+/[0-9]+ \[[^]]*\]" | tail -1
done
echo
echo "Per-split file counts:"
for split in train val test; do
    c=$(find "$EXTRACT_DIR/$split" -name "*.npy" 2>/dev/null | wc -l)
    echo "  $split: $c files"
done
echo
err=$(grep -c "ERROR" "$TASK_LOG" 2>/dev/null)
echo "Extraction errors: ${err:-0}"
echo

echo "=== Training progress (if started) ==="
if [ -f "$TRAIN_LOG" ]; then
    epochs=$(grep -cE "^Epoch +[0-9]+/" "$TRAIN_LOG" 2>/dev/null)
    echo "Epochs done: $epochs"
    echo "Latest epoch line:"
    grep -E "^Epoch +[0-9]+/" "$TRAIN_LOG" 2>/dev/null | tail -1
    echo "Latest EMA+TTA:"
    grep "EMA+TTA" "$TRAIN_LOG" 2>/dev/null | tail -1
    echo "Latest tqdm:"
    tail -200 "$TRAIN_TASK_LOG" 2>/dev/null | grep -oE "E[0-9]+/[0-9]+ (train|val)\s*: +[0-9]+%.*\| [0-9]+/[0-9]+ \[[^]]*\]" | tail -1
else
    echo "(training hasn't started yet)"
fi
echo

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader

echo
echo "=== Disk ==="
df -h /c | tail -1
