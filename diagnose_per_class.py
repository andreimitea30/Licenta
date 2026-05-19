import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default="v4", choices=["v4", "v5"],
                    help="Which v* module to use (default: v4)")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to checkpoint .pth (default: best_stgcn_<model>_emattta.pth)")
    ap.add_argument("--csv",        default=None,
                    help="Output CSV path (default: per_class_accuracy_<model>.csv)")
    ap.add_argument("--splits", default="val,test",
                    help="Comma-separated splits to evaluate on (default: val,test)")
    ap.add_argument("--no-tta", action="store_true",
                    help="Disable horizontal-flip TTA at eval (default: TTA on)")
    ap.add_argument("--bottom", type=int, default=50, help="how many worst classes to print")
    ap.add_argument("--top",    type=int, default=20, help="how many best classes to print")
    args = ap.parse_args()

    import importlib
    mod = importlib.import_module(f"st_gcn_{args.model}")

    if args.checkpoint is None:
        candidates = [
            ROOT / f"best_stgcn_{args.model}_emattta.pth",
            ROOT / f"best_stgcn_{args.model}.pth",
        ]
        args.checkpoint = next((str(c) for c in candidates if c.exists()), str(candidates[0]))
    if args.csv is None:
        args.csv = str(ROOT / f"per_class_accuracy_{args.model}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model module: st_gcn_{args.model}")
    print(f"Checkpoint: {args.checkpoint}")

    train_ds = mod.HAA500Dataset("train", augment=False)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"Train classes: {len(class_to_idx)}")

    eval_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"Eval splits: {eval_splits}")
    eval_datasets = []
    for s in eval_splits:
        ds = mod.HAA500Dataset(s, class_to_idx=class_to_idx, augment=False)
        eval_datasets.append((s, ds))
        print(f"  {s}: {len(ds)} samples")

    model = mod.ThreeStreamSTGCN(num_classes=500).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")

    use_tta = not args.no_tta
    print(f"TTA: {'ON (horizontal flip)' if use_tta else 'OFF'}")

    records = []
    for split_name, ds in eval_datasets:
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=False)
        with torch.no_grad():
            for joint, bone, motion, y in loader:
                joint  = joint.to(device,  non_blocking=True)
                bone   = bone.to(device,   non_blocking=True)
                motion = motion.to(device, non_blocking=True)
                y      = y.to(device,      non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(joint, bone, motion)
                    if use_tta:
                        j2, b2, m2 = mod._hflip_streams(joint, bone, motion)
                        logits2 = model(j2, b2, m2)
                        probs = (logits.softmax(-1) + logits2.softmax(-1)) / 2
                    else:
                        probs = logits.softmax(-1)
                top1 = probs.argmax(1)
                top5 = probs.topk(5, 1).indices
                for i in range(y.size(0)):
                    records.append((int(y[i].item()), int(top1[i].item()),
                                    [int(x) for x in top5[i].tolist()]))
        print(f"  {split_name}: {len(ds)} done")

    n = len(records)
    overall_t1 = 100 * sum(1 for t, p1, _ in records if p1 == t) / n
    overall_t5 = 100 * sum(1 for t, _, t5 in records if t in t5) / n
    print(f"\n=== OVERALL on val+test ===")
    print(f"  Samples: {n}")
    print(f"  Top-1: {overall_t1:.2f}%")
    print(f"  Top-5: {overall_t5:.2f}%")

    per_class = defaultdict(lambda: {"t1": 0, "t5": 0, "n": 0,
                                      "confusions": defaultdict(int)})
    for true, pred1, top5 in records:
        c = per_class[true]
        c["n"] += 1
        if pred1 == true:
            c["t1"] += 1
        if true in top5:
            c["t5"] += 1
        if pred1 != true:
            c["confusions"][pred1] += 1

    rows = []
    for cls_idx in sorted(per_class.keys()):
        c = per_class[cls_idx]
        confs = sorted(c["confusions"].items(), key=lambda kv: -kv[1])
        top_confusions = ", ".join(f"{idx_to_class[ci]}({cnt})" for ci, cnt in confs[:3])
        rows.append({
            "class": idx_to_class[cls_idx],
            "idx": cls_idx,
            "top1_acc":   100.0 * c["t1"] / c["n"] if c["n"] else 0.0,
            "top5_acc":   100.0 * c["t5"] / c["n"] if c["n"] else 0.0,
            "samples":    c["n"],
            "top_conf_1": idx_to_class[confs[0][0]] if confs else "",
            "top_conf_1_count": confs[0][1] if confs else 0,
            "top_confusions": top_confusions,
        })

    bins = [0, 16.7, 33.4, 50, 66.7, 83.4, 100.0]
    bin_labels = ["0%", "<33%", "<50%", "<67%", "<84%", "<100%", "100%"]
    counts = [0] * len(bin_labels)
    for r in rows:
        a = r["top1_acc"]
        if a == 0.0: counts[0] += 1
        elif a < 33.4: counts[1] += 1
        elif a < 50.0: counts[2] += 1
        elif a < 66.7: counts[3] += 1
        elif a < 83.4: counts[4] += 1
        elif a < 100.0: counts[5] += 1
        else: counts[6] += 1

    print(f"\n=== PER-CLASS Top-1 DISTRIBUTION (500 classes total) ===")
    for label, count in zip(bin_labels, counts):
        bar = "#" * (count // 4)
        print(f"  {label:>5}  | {count:>3} classes  {bar}")

    rows.sort(key=lambda r: (r["top1_acc"], r["top5_acc"]))
    print(f"\n=== BOTTOM {args.bottom} CLASSES (worst Top-1) ===")
    print(f"{'class':<35} {'top1':>5} {'top5':>5} {'n':>3}  most confused with (count)")
    print("-" * 100)
    for r in rows[:args.bottom]:
        print(f"{r['class']:<35} {r['top1_acc']:>4.0f}% {r['top5_acc']:>4.0f}% {r['samples']:>3}  {r['top_confusions']}")

    print(f"\n=== TOP {args.top} CLASSES (best Top-1) ===")
    print(f"{'class':<35} {'top1':>5} {'top5':>5} {'n':>3}")
    print("-" * 60)
    for r in rows[-args.top:][::-1]:
        print(f"{r['class']:<35} {r['top1_acc']:>4.0f}% {r['top5_acc']:>4.0f}% {r['samples']:>3}")

    multi_keywords = [
        "shake_hands", "shaking", "hug", "kiss", "fight", "wrestle",
        "boxing", "fencing", "martial", "judo", "karate", "kung_fu",
        "mma", "sumo", "sparring", "punch", "kick_box", "tae_kwon",
        "dance", "tango", "salsa", "waltz", "ballroom", "swing_dance",
        "doubles", "pair", "couple", "high_five", "fist_bump",
        "carry", "lift_person", "piggy", "duel", "spar",
    ]
    def is_multi_person(name: str) -> bool:
        low = name.lower()
        return any(k in low for k in multi_keywords)

    multi_rows = [r for r in rows if is_multi_person(r["class"])]
    other_rows = [r for r in rows if not is_multi_person(r["class"])]
    if multi_rows:
        m_t1 = sum(r["top1_acc"] * r["samples"] for r in multi_rows) / sum(r["samples"] for r in multi_rows)
        o_t1 = sum(r["top1_acc"] * r["samples"] for r in other_rows) / sum(r["samples"] for r in other_rows)
        m_t5 = sum(r["top5_acc"] * r["samples"] for r in multi_rows) / sum(r["samples"] for r in multi_rows)
        o_t5 = sum(r["top5_acc"] * r["samples"] for r in other_rows) / sum(r["samples"] for r in other_rows)
        print(f"\n=== HEURISTIC: keyword-flagged 'likely multi-person' classes ===")
        print(f"  Flagged classes: {len(multi_rows)} / {len(rows)}")
        print(f"  Sample-weighted Top-1  flagged={m_t1:.1f}%  others={o_t1:.1f}%  delta={m_t1-o_t1:+.1f}pp")
        print(f"  Sample-weighted Top-5  flagged={m_t5:.1f}%  others={o_t5:.1f}%  delta={m_t5-o_t5:+.1f}pp")
        print(f"  (Heuristic only — see CSV / bottom list for ground truth.)")
        print(f"\n  Flagged classes by Top-1 (worst -> best):")
        for r in sorted(multi_rows, key=lambda r: r["top1_acc"]):
            print(f"    {r['class']:<35} top1={r['top1_acc']:>4.0f}%  top5={r['top5_acc']:>4.0f}%  {r['top_confusions']}")

    csv_path = Path(args.csv)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nFull per-class report: {csv_path}")

if __name__ == "__main__":
    main()
