"""Run st_gcn_v2 and st_gcn_v3 head-to-head for a few epochs and print a side-by-side comparison.

Existing best_stgcn_v2.pth / checkpoint_v2.pth are NOT touched: each model writes to a
dedicated _compare_* file under the project root.

Per-epoch summary lines from each train() are streamed to dedicated *.log files so
output is preserved even if a run crashes.
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
    """Write to multiple streams; flush each line immediately so the log survives crashes."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def run_one(label: str, train_fn, num_epochs: int, ckpt: Path, best: Path,
            seed: int, log_path: Path):
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

    rows = []
    for m in EPOCH_RE.finditer(log_path.read_text(encoding="utf-8")):
        rows.append({
            "epoch": int(m.group(1)),
            "lr": float(m.group(2)),
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
    args = ap.parse_args()

    paths = {
        "v2_ckpt": ROOT / "checkpoint_v2_compare.pth",
        "v2_best": ROOT / "best_stgcn_v2_compare.pth",
        "v3_ckpt": ROOT / "checkpoint_v3_compare.pth",
        "v3_best": ROOT / "best_stgcn_v3_compare.pth",
        "v2_log":  ROOT / "compare_v2.log",
        "v3_log":  ROOT / "compare_v3.log",
    }
    for k, p in paths.items():
        if k.endswith("_ckpt") or k.endswith("_best"):
            if p.exists():
                p.unlink()

    import st_gcn_v2 as v2
    import st_gcn_v3 as v3

    v2_rows = run_one("v2", v2.train, args.epochs, paths["v2_ckpt"], paths["v2_best"],
                      args.seed, paths["v2_log"])
    v3_rows = run_one("v3", v3.train, args.epochs, paths["v3_ckpt"], paths["v3_best"],
                      args.seed, paths["v3_log"])

    print("\n=== SIDE-BY-SIDE ===")
    header = (f"{'Epoch':>5} | {'v2 TrLoss':>9} {'v2 TrAcc':>8} {'v2 Top1':>7} {'v2 Top5':>7} "
              f"|| {'v3 TrLoss':>9} {'v3 TrAcc':>8} {'v3 Top1':>7} {'v3 Top5':>7}")
    print(header)
    print("-" * len(header))
    for i in range(min(len(v2_rows), len(v3_rows))):
        a, b = v2_rows[i], v3_rows[i]
        print(f"{a['epoch']:>5} | "
              f"{a['train_loss']:>9.4f} {a['train_acc']:>7.1f}% {a['val_top1']:>6.1f}% {a['val_top5']:>6.1f}% "
              f"|| {b['train_loss']:>9.4f} {b['train_acc']:>7.1f}% {b['val_top1']:>6.1f}% {b['val_top5']:>6.1f}%")

    if v2_rows and v3_rows:
        last_v2, last_v3 = v2_rows[-1], v3_rows[-1]
        print(f"\nFinal-epoch deltas (v3 - v2): "
              f"Top-1 {last_v3['val_top1'] - last_v2['val_top1']:+.2f}pp, "
              f"Top-5 {last_v3['val_top5'] - last_v2['val_top5']:+.2f}pp")


if __name__ == "__main__":
    main()
