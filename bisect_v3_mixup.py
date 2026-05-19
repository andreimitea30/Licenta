import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

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
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()

def parse_log(path: Path):
    rows = []
    if not path.exists():
        return rows
    for m in EPOCH_RE.finditer(path.read_text(encoding="utf-8")):
        rows.append({
            "epoch": int(m.group(1)),
            "train_loss": float(m.group(3)),
            "train_acc": float(m.group(4)),
            "val_top1": float(m.group(5)),
            "val_top5": float(m.group(6)),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--alpha",  type=float, default=0.3)
    args = ap.parse_args()

    ckpt = ROOT / "checkpoint_v3_mixup_compare.pth"
    best = ROOT / "best_stgcn_v3_mixup_compare.pth"
    log  = ROOT / "compare_v3_mixup.log"
    for p in (ckpt, best):
        if p.exists(): p.unlink()

    import st_gcn_v3 as v3
    v3.MIXUP_ALPHA = args.alpha
    print(f"v3.MIXUP_ALPHA overridden -> {v3.MIXUP_ALPHA}")

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

    v2 = parse_log(ROOT / "compare_v2.log")
    v3_no = parse_log(ROOT / "compare_v3.log")
    v3_mx = parse_log(log)

    print("\n=== 3-WAY: v2 vs v3 (no mixup) vs v3+mixup ===")
    h = (f"{'E':>2} | {'v2 TrL':>6} {'v2 T1':>5} {'v2 T5':>5} || "
         f"{'v3 TrL':>6} {'v3 T1':>5} {'v3 T5':>5} || "
         f"{'v3m TrL':>7} {'v3m T1':>6} {'v3m T5':>6}")
    print(h); print("-" * len(h))
    n = max(len(v2), len(v3_no), len(v3_mx))
    for i in range(n):
        a = v2[i]    if i < len(v2)    else None
        b = v3_no[i] if i < len(v3_no) else None
        c = v3_mx[i] if i < len(v3_mx) else None
        def f(r, key, w, suf=""):
            return f"{'  -':>{w}}" if r is None else f"{r[key]:>{w}.2f}{suf}"
        print(f"{i+1:>2} | "
              f"{f(a, 'train_loss', 6)} {f(a, 'val_top1', 5)} {f(a, 'val_top5', 5)} || "
              f"{f(b, 'train_loss', 6)} {f(b, 'val_top1', 5)} {f(b, 'val_top5', 5)} || "
              f"{f(c, 'train_loss', 7)} {f(c, 'val_top1', 6)} {f(c, 'val_top5', 6)}")

    if v2 and v3_mx:
        a, c = v2[-1], v3_mx[-1]
        print(f"\nFinal-epoch v3+mixup vs v2: Top-1 {c['val_top1']-a['val_top1']:+.2f}pp, "
              f"Top-5 {c['val_top5']-a['val_top5']:+.2f}pp")
    if v3_no and v3_mx:
        b, c = v3_no[-1], v3_mx[-1]
        print(f"Final-epoch v3+mixup vs v3 no-mixup: Top-1 {c['val_top1']-b['val_top1']:+.2f}pp, "
              f"Top-5 {c['val_top5']-b['val_top5']:+.2f}pp")

if __name__ == "__main__":
    main()
