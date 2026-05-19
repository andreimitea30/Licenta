import argparse
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

os.environ["V3_AUGMENT_MODE"] = "v2"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*LR\s+([\d.eE+-]+)\s*\|\s*"
    r"Train\s+([\d.]+)/([\d.]+)%\s*\|\s*"
    r"Val\s+Top-1\s+([\d.]+)%\s+Top-5\s+([\d.]+)%"
)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()

def parse_log(path: Path):
    rows = []
    if not path.exists(): return rows
    for m in EPOCH_RE.finditer(path.read_text(encoding="utf-8")):
        rows.append({"epoch": int(m.group(1)),
                     "train_acc": float(m.group(4)),
                     "val_top1":  float(m.group(5)),
                     "val_top5":  float(m.group(6))})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    ckpt = ROOT / "checkpoint_v3_bisectC.pth"
    best = ROOT / "best_stgcn_v3_bisectC.pth"
    log  = ROOT / "compare_v3_bisectC_v2aug.log"
    for p in (ckpt, best):
        if p.exists(): p.unlink()

    import st_gcn_v3 as v3
    print(f"V3_AUGMENT_MODE env = {os.environ.get('V3_AUGMENT_MODE')}")
    print(f"v3 settings: DROPOUT={v3.DROPOUT}  MIXUP_ALPHA={v3.MIXUP_ALPHA}  "
          f"ATTENTION_BLOCKS={v3.ATTENTION_BLOCKS}  USE_WEIGHTED_SAMPLER={v3.USE_WEIGHTED_SAMPLER}")
    set_seed(args.seed)

    real_stdout = sys.stdout
    with open(log, "w", encoding="utf-8", buffering=1) as f:
        sys.stdout = Tee(real_stdout, f)
        try:
            v3.train(num_epochs=args.epochs,
                     checkpoint_path=str(ckpt),
                     model_save_path=str(best))
        except Exception:
            traceback.print_exc(file=sys.stdout)
        finally:
            sys.stdout = real_stdout

    logs = {
        'v2 baseline'         : ROOT / 'compare_v2.log',
        'v3 hi-reg'           : ROOT / 'compare_v3_hireg.log',
        'Bisect A (shuffle)'  : ROOT / 'compare_v3_bisectA_shuffle.log',
        'Bisect B (attn 8,9)' : ROOT / 'compare_v3_bisectB_attn89.log',
        'Bisect C (v2 aug)'   : log,
    }
    print("\n=== 5-WAY SUMMARY (Best/Last-5-Mean across 30 epochs) ===")
    print(f"{'Variant':<22} | {'Best Top-1':>11} (ep) | {'Best Top-5':>11} (ep) | {'Last-5 mean T1':>14} | {'Last-5 mean T5':>14} | {'gap':>5}")
    print('-' * 110)
    for name, file in logs.items():
        rows = parse_log(file)
        if not rows: continue
        best_t1 = max(rows, key=lambda r: r['val_top1'])
        best_t5 = max(rows, key=lambda r: r['val_top5'])
        last5_t1 = sum(r['val_top1'] for r in rows[-5:]) / 5
        last5_t5 = sum(r['val_top5'] for r in rows[-5:]) / 5
        gap = rows[-1]['train_acc'] - rows[-1]['val_top1']
        print(f'{name:<22} | {best_t1["val_top1"]:>9.1f}%  ({best_t1["epoch"]:>2}) | '
              f'{best_t5["val_top5"]:>9.1f}%  ({best_t5["epoch"]:>2}) | '
              f'{last5_t1:>13.1f}% | {last5_t5:>13.1f}% | {gap:>4.1f}pp')

if __name__ == "__main__":
    main()
