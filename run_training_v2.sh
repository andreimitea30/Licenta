#!/bin/bash
source /home/andrei/Licenta/Licenta/venv_linux/bin/activate
cd /home/andrei/Licenta/Licenta

TOTAL=10000
LOG=/home/andrei/Licenta/Licenta/training_v2.log

echo "[$(date)] Waiting for 3D extraction to complete..." | tee -a "$LOG"

while true; do
    COUNT=$(find extracted_skeletons_world/ -name "*.npy" 2>/dev/null | wc -l)
    echo "[$(date)] Extracted: $COUNT / $TOTAL" | tee -a "$LOG"
    if [ "$COUNT" -ge "$TOTAL" ]; then
        echo "[$(date)] Extraction complete. Starting training." | tee -a "$LOG"
        break
    fi
    sleep 300
done

python src/st_gcn_v2.py 2>&1 | tee -a "$LOG"
echo "[$(date)] Training finished." | tee -a "$LOG"

echo "[$(date)] Finalizing thesis with Run 2 results..." | tee -a "$LOG"
python thesis/finalize_thesis.py 2>&1 | tee -a "$LOG"
