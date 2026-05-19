"""Train v4 with EMA + TTA on for 50 epochs and present a 4-way ablation:
   live | live+TTA | EMA | EMA+TTA — all extracted from a single training run.

Compares against compare_v2_v4_v4.log (the v4-baseline overnight run, no EMA / no TTA).
"""
import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Live no-TTA line (same regex as the compare_v2_v4 driver — kept for backward compat)
LIVE_RE     = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*LR\s+([\d.eE+-]+)\s*\|\s*"
    r"Train\s+([\d.]+)/([\d.]+)%\s*\|\s*"
    r"Val\s+Top-1\s+([\d.]+)%\s+Top-5\s+([\d.]+)%"
)
LIVE_TTA_RE = re.compile(r"Live\+TTA\s+Top-1\s+([\d.]+)%\s+Top-5\s+([\d.]+)%")
EMA_RE      = re.compile(r"EMA\s+Top-1\s+([\d.]+)%\s+Top-5\s+([\d.]+)%")
EMA_TTA_RE  = re.compile(r"EMA\+TTA\s+Top-1\s+([\d.]+)%\s+Top-5\s+([\d.]+)%")


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
    """Return list of dicts, one per epoch, with all 4 metric variants when present."""
    if not path.exists(): return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows = []
    cur = None
    for ln in lines:
        m = LIVE_RE.match(ln)
        if m:
            if cur: rows.append(cur)
            cur = {
                "epoch": int(m.group(1)),
                "live_t1": float(m.group(5)),
                "live_t5": float(m.group(6)),
            }
            continue
        if cur is None: continue
        m = LIVE_TTA_RE.search(ln)
        if m:
            cur["live_tta_t1"] = float(m.group(1)); cur["live_tta_t5"] = float(m.group(2)); continue
        m = EMA_TTA_RE.search(ln)
        if m:
            cur["ema_tta_t1"] = float(m.group(1)); cur["ema_tta_t5"] = float(m.group(2)); continue
        m = EMA_RE.search(ln)
        if m:
            cur["ema_t1"] = float(m.group(1)); cur["ema_t5"] = float(m.group(2)); continue
    if cur: rows.append(cur)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    ckpt = ROOT / "checkpoint_v4_emattta.pth"
    best = ROOT / "best_stgcn_v4_emattta.pth"
    log  = ROOT / "compare_v4_emattta.log"
    for p in (ckpt, best):
        if p.exists(): p.unlink()

    import st_gcn_v4 as v4
    print(f"v4: USE_EMA={v4.USE_EMA}  USE_TTA={v4.USE_TTA}  EMA_DECAY={v4.EMA_DECAY}")
    set_seed(args.seed)

    real_stdout = sys.stdout
    with open(log, "w", encoding="utf-8", buffering=1) as f:
        sys.stdout = Tee(real_stdout, f)
        try:
            v4.train(num_epochs=args.epochs,
                     checkpoint_path=str(ckpt),
                     model_save_path=str(best))
        except Exception:
            traceback.print_exc(file=sys.stdout)
        finally:
            sys.stdout = real_stdout

    base_rows = parse_log(ROOT / "compare_v2_v4_v4.log")  # v4 baseline (no EMA / no TTA)
    new_rows  = parse_log(log)

    print("\n=== 4-WAY ABLATION (single training run) ===")
    h = (f"{'E':>3} | {'live T1':>7} {'live T5':>7} || "
         f"{'+TTA T1':>7} {'+TTA T5':>7} || "
         f"{'EMA T1':>6} {'EMA T5':>6} || "
         f"{'E+T T1':>6} {'E+T T5':>6}")
    print(h); print("-" * len(h))
    for r in new_rows:
        def f(d, key, w):
            return f"{'-':>{w}}" if key not in d else f"{d[key]:>{w}.1f}"
        print(f"{r['epoch']:>3} | "
              f"{f(r,'live_t1',7)} {f(r,'live_t5',7)} || "
              f"{f(r,'live_tta_t1',7)} {f(r,'live_tta_t5',7)} || "
              f"{f(r,'ema_t1',6)} {f(r,'ema_t5',6)} || "
              f"{f(r,'ema_tta_t1',6)} {f(r,'ema_tta_t5',6)}")

    print("\n=== ROBUST METRICS (best-across-epochs / last-5-mean) ===")
    print(f"{'Variant':<22} | {'Best Top-1':>11} (ep) | {'Best Top-5':>11} (ep) | {'Last-5 mean T1':>14} | {'Last-5 mean T5':>14}")
    print('-' * 105)
    def stat(rows, t1key, t5key, label):
        vals_t1 = [(r['epoch'], r[t1key]) for r in rows if t1key in r]
        vals_t5 = [(r['epoch'], r[t5key]) for r in rows if t5key in r]
        if not vals_t1: return
        bt1 = max(vals_t1, key=lambda x: x[1])
        bt5 = max(vals_t5, key=lambda x: x[1])
        last5_t1 = sum(v for _, v in vals_t1[-5:]) / min(5, len(vals_t1))
        last5_t5 = sum(v for _, v in vals_t5[-5:]) / min(5, len(vals_t5))
        print(f"{label:<22} | {bt1[1]:>9.1f}%  ({bt1[0]:>2}) | {bt5[1]:>9.1f}%  ({bt5[0]:>2}) | {last5_t1:>13.1f}% | {last5_t5:>13.1f}%")

    if base_rows:
        stat(base_rows, 'live_t1', 'live_t5', 'v4 baseline (overnight)')
    stat(new_rows, 'live_t1',     'live_t5',     'v4 live (this run)')
    stat(new_rows, 'live_tta_t1', 'live_tta_t5', 'v4 + TTA')
    stat(new_rows, 'ema_t1',      'ema_t5',      'v4 + EMA')
    stat(new_rows, 'ema_tta_t1',  'ema_tta_t5',  'v4 + EMA + TTA')


if __name__ == "__main__":
    main()
