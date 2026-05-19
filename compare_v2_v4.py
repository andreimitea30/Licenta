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
                     "lr": float(m.group(2)),
                     "train_loss": float(m.group(3)),
                     "train_acc": float(m.group(4)),
                     "val_top1": float(m.group(5)),
                     "val_top5": float(m.group(6))})
    return rows

def run_one(label, train_fn, num_epochs, ckpt, best, seed, log_path):
    for p in (ckpt, best):
        if p.exists(): p.unlink()
    set_seed(seed)
    print(f"\n=== Running {label} for {num_epochs} epochs (seed={seed}) ===", flush=True)
    real_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8", buffering=1) as f:
        sys.stdout = Tee(real_stdout, f)
        try:
            train_fn(num_epochs=num_epochs,
                     checkpoint_path=str(ckpt),
                     model_save_path=str(best))
        except Exception:
            traceback.print_exc(file=sys.stdout)
        finally:
            sys.stdout = real_stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    paths = {
        "v2_ckpt": ROOT / "checkpoint_v2_v4cmp.pth",
        "v2_best": ROOT / "best_stgcn_v2_v4cmp.pth",
        "v4_ckpt": ROOT / "checkpoint_v4_v4cmp.pth",
        "v4_best": ROOT / "best_stgcn_v4_v4cmp.pth",
        "v2_log":  ROOT / "compare_v2_v4_v2.log",
        "v4_log":  ROOT / "compare_v2_v4_v4.log",
    }

    import st_gcn_v2 as v2
    run_one("v2 (2-stream baseline)", v2.train, args.epochs,
            paths["v2_ckpt"], paths["v2_best"], args.seed, paths["v2_log"])

    import st_gcn_v4 as v4
    run_one("v4 (3-stream: joint + bone + motion)", v4.train, args.epochs,
            paths["v4_ckpt"], paths["v4_best"], args.seed, paths["v4_log"])

    v2_rows = parse_log(paths["v2_log"])
    v4_rows = parse_log(paths["v4_log"])

    print("\n=== SIDE-BY-SIDE: v2 vs v4 ===")
    h = (f"{'E':>3} | {'v2 TrLoss':>9} {'v2 TrAcc':>8} {'v2 Top1':>7} {'v2 Top5':>7} || "
         f"{'v4 TrLoss':>9} {'v4 TrAcc':>8} {'v4 Top1':>7} {'v4 Top5':>7}")
    print(h); print("-" * len(h))
    n = max(len(v2_rows), len(v4_rows))
    for i in range(n):
        a = v2_rows[i] if i < len(v2_rows) else None
        b = v4_rows[i] if i < len(v4_rows) else None
        def f(r, key, w, suf=""):
            return f"{'-':>{w}}" if r is None else f"{r[key]:>{w-len(suf)}.4f}{suf}" if 'loss' in key else f"{r[key]:>{w-len(suf)}.1f}{suf}"
        print(f"{(i+1):>3} | "
              f"{f(a,'train_loss',9)} {f(a,'train_acc',7,'%')} {f(a,'val_top1',6,'%')} {f(a,'val_top5',6,'%')} || "
              f"{f(b,'train_loss',9)} {f(b,'train_acc',7,'%')} {f(b,'val_top1',6,'%')} {f(b,'val_top5',6,'%')}")

    print("\n=== ROBUST METRICS ===")
    print(f"{'Variant':<8} | {'Best Top-1':>11} (ep) | {'Best Top-5':>11} (ep) | {'Last-5 mean T1':>14} | {'Last-5 mean T5':>14} | {'final gap':>10}")
    print('-' * 105)
    for name, rows in [("v2", v2_rows), ("v4", v4_rows)]:
        if not rows: continue
        best_t1 = max(rows, key=lambda r: r['val_top1'])
        best_t5 = max(rows, key=lambda r: r['val_top5'])
        last5_t1 = sum(r['val_top1'] for r in rows[-5:]) / min(5, len(rows))
        last5_t5 = sum(r['val_top5'] for r in rows[-5:]) / min(5, len(rows))
        gap = rows[-1]['train_acc'] - rows[-1]['val_top1']
        print(f"{name:<8} | {best_t1['val_top1']:>9.1f}%  ({best_t1['epoch']:>2}) | "
              f"{best_t5['val_top5']:>9.1f}%  ({best_t5['epoch']:>2}) | "
              f"{last5_t1:>13.1f}% | {last5_t5:>13.1f}% | {gap:>8.1f}pp")

    if v2_rows and v4_rows:
        a, b = v2_rows[-1], v4_rows[-1]
        print(f"\nFinal-epoch deltas (v4 - v2): "
              f"Top-1 {b['val_top1']-a['val_top1']:+.2f}pp, "
              f"Top-5 {b['val_top5']-a['val_top5']:+.2f}pp")
        ba = max(v2_rows, key=lambda r: r['val_top1'])['val_top1']
        bb = max(v4_rows, key=lambda r: r['val_top1'])['val_top1']
        print(f"Best-Top-1 deltas    (v4 - v2): {bb - ba:+.2f}pp")

if __name__ == "__main__":
    main()
