"""Two targeted bisects of v3 architectural changes (regularization restored: DROPOUT=0.5, mixup=0.3).

Bisect A: revert sampler to plain shuffle=True (keep distributed attention [3,6,9]).
Bisect B: revert attention to v2 placement [8,9]   (keep weighted sampler).

Compare against existing 30-epoch logs from prior runs:
  compare_v2.log         -> v2 baseline
  compare_v3_hireg.log   -> v3 with all changes (high-reg)
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
        rows.append({"epoch": int(m.group(1)),
                     "train_acc":  float(m.group(4)),
                     "val_top1":   float(m.group(5)),
                     "val_top5":   float(m.group(6))})
    return rows


def run_bisect(label, attention_blocks, use_weighted_sampler,
               num_epochs, seed, log_path, ckpt, best):
    for p in (ckpt, best):
        if p.exists(): p.unlink()

    # Reload v3 fresh for each bisect so toggles take effect on a clean module state
    if "st_gcn_v3" in sys.modules:
        del sys.modules["st_gcn_v3"]
    import st_gcn_v3 as v3
    v3.ATTENTION_BLOCKS     = attention_blocks
    v3.USE_WEIGHTED_SAMPLER = use_weighted_sampler

    print(f"\n=== {label} ===")
    print(f"v3 settings: ATTENTION_BLOCKS={v3.ATTENTION_BLOCKS}  "
          f"USE_WEIGHTED_SAMPLER={v3.USE_WEIGHTED_SAMPLER}  "
          f"DROPOUT={v3.DROPOUT}  MIXUP_ALPHA={v3.MIXUP_ALPHA}", flush=True)
    set_seed(seed)

    real_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8", buffering=1) as f:
        sys.stdout = Tee(real_stdout, f)
        try:
            v3.train(num_epochs=num_epochs,
                     checkpoint_path=str(ckpt),
                     model_save_path=str(best))
        except Exception:
            traceback.print_exc(file=sys.stdout)
        finally:
            sys.stdout = real_stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    log_a = ROOT / "compare_v3_bisectA_shuffle.log"
    log_b = ROOT / "compare_v3_bisectB_attn89.log"

    run_bisect("BISECT A — shuffle=True, attention=[3,6,9]",
               attention_blocks=[3, 6, 9], use_weighted_sampler=False,
               num_epochs=args.epochs, seed=args.seed, log_path=log_a,
               ckpt=ROOT / "checkpoint_v3_bisectA.pth",
               best=ROOT / "best_stgcn_v3_bisectA.pth")

    run_bisect("BISECT B — weighted sampler, attention=[8,9]",
               attention_blocks=[8, 9], use_weighted_sampler=True,
               num_epochs=args.epochs, seed=args.seed, log_path=log_b,
               ckpt=ROOT / "checkpoint_v3_bisectB.pth",
               best=ROOT / "best_stgcn_v3_bisectB.pth")

    v2  = parse_log(ROOT / "compare_v2.log")
    hi  = parse_log(ROOT / "compare_v3_hireg.log")
    a   = parse_log(log_a)
    b   = parse_log(log_b)

    print("\n=== 30-EPOCH 4-WAY ===")
    h = (f"{'E':>2} | "
         f"{'v2 T1':>5} {'v2 T5':>5} || "
         f"{'hi T1':>5} {'hi T5':>5} || "
         f"{'A T1':>5} {'A T5':>5} || "
         f"{'B T1':>5} {'B T5':>5}")
    print(h); print("-" * len(h))
    n = max(len(v2), len(hi), len(a), len(b))
    for i in range(n):
        def f(rows, key, w):
            if i >= len(rows): return f"{'-':>{w}}"
            return f"{rows[i][key]:>{w}.1f}"
        print(f"{i+1:>2} | "
              f"{f(v2,'val_top1',5)} {f(v2,'val_top5',5)} || "
              f"{f(hi,'val_top1',5)} {f(hi,'val_top5',5)} || "
              f"{f(a,'val_top1',5)} {f(a,'val_top5',5)} || "
              f"{f(b,'val_top1',5)} {f(b,'val_top5',5)}")

    print("\nFinal-epoch deltas vs v2:")
    for label, rows in [("v3 hi-reg     ", hi), ("Bisect A (shuf)", a), ("Bisect B (8,9) ", b)]:
        if rows and v2:
            print(f"  {label}: Top-1 {rows[-1]['val_top1']-v2[-1]['val_top1']:+.2f}pp,  "
                  f"Top-5 {rows[-1]['val_top5']-v2[-1]['val_top5']:+.2f}pp,  "
                  f"train-val gap {rows[-1]['train_acc']-rows[-1]['val_top1']:.1f}pp")

    print("\nFinal-epoch deltas vs v3 hi-reg (isolating each architectural component):")
    for label, rows in [("Bisect A (shuf)", a), ("Bisect B (8,9) ", b)]:
        if rows and hi:
            print(f"  {label}: Top-1 {rows[-1]['val_top1']-hi[-1]['val_top1']:+.2f}pp,  "
                  f"Top-5 {rows[-1]['val_top5']-hi[-1]['val_top5']:+.2f}pp")


if __name__ == "__main__":
    main()
