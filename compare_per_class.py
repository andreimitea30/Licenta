import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read_csv(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ROOT / "per_class_accuracy_v4.csv"))
    ap.add_argument("--new",      default=str(ROOT / "per_class_accuracy_v5.csv"))
    ap.add_argument("--out",      default=str(ROOT / "compare_v4_v5.csv"))
    ap.add_argument("--bottom-new", type=int, default=30)
    ap.add_argument("--top-improved", type=int, default=30)
    ap.add_argument("--top-regressed", type=int, default=15)
    args = ap.parse_args()

    base = {r["class"]: r for r in read_csv(Path(args.baseline))}
    new  = {r["class"]: r for r in read_csv(Path(args.new))}
    common = sorted(set(base) & set(new))
    print(f"Baseline rows: {len(base)}  New rows: {len(new)}  Common: {len(common)}")

    rows = []
    for c in common:
        b = base[c]; n = new[c]
        rows.append({
            "class": c,
            "baseline_top1": float(b["top1_acc"]),
            "new_top1":      float(n["top1_acc"]),
            "delta_top1":    float(n["top1_acc"]) - float(b["top1_acc"]),
            "baseline_top5": float(b["top5_acc"]),
            "new_top5":      float(n["top5_acc"]),
            "delta_top5":    float(n["top5_acc"]) - float(b["top5_acc"]),
            "samples":       int(n["samples"]),
            "baseline_top_conf": b.get("top_conf_1", ""),
            "new_top_conf":      n.get("top_conf_1", ""),
        })

    n_total = sum(r["samples"] for r in rows)
    base_t1 = sum(r["baseline_top1"] * r["samples"] for r in rows) / n_total
    new_t1  = sum(r["new_top1"]      * r["samples"] for r in rows) / n_total
    base_t5 = sum(r["baseline_top5"] * r["samples"] for r in rows) / n_total
    new_t5  = sum(r["new_top5"]      * r["samples"] for r in rows) / n_total

    print(f"\n=== OVERALL (sample-weighted) ===")
    print(f"  Top-1: baseline={base_t1:.2f}%  new={new_t1:.2f}%  delta={new_t1-base_t1:+.2f}pp")
    print(f"  Top-5: baseline={base_t5:.2f}%  new={new_t5:.2f}%  delta={new_t5-base_t5:+.2f}pp")

    improved   = sum(1 for r in rows if r["delta_top1"] > 0)
    regressed  = sum(1 for r in rows if r["delta_top1"] < 0)
    unchanged  = sum(1 for r in rows if r["delta_top1"] == 0)
    big_improve = sum(1 for r in rows if r["delta_top1"] >= 33.0)
    big_regress = sum(1 for r in rows if r["delta_top1"] <= -33.0)
    print(f"\n=== CLASS-LEVEL MOVEMENT ===")
    print(f"  Improved:   {improved}   ({big_improve} by >=33pp)")
    print(f"  Regressed:  {regressed}  ({big_regress} by >=33pp)")
    print(f"  Unchanged:  {unchanged}")

    rows_by_new = sorted(rows, key=lambda r: (r["new_top1"], r["new_top5"]))
    rows_by_imp = sorted(rows, key=lambda r: -r["delta_top1"])
    rows_by_reg = sorted(rows, key=lambda r: r["delta_top1"])

    print(f"\n=== STILL FAILING (bottom {args.bottom_new} by new Top-1) ===")
    print(f"{'class':<35} {'base':>5} -> {'new':>5}  d {'top1':>5}  ({'top5 base->new':>14})")
    print("-" * 100)
    for r in rows_by_new[:args.bottom_new]:
        print(f"{r['class']:<35} {r['baseline_top1']:>4.0f}% -> {r['new_top1']:>4.0f}%  "
              f"d {r['delta_top1']:>+5.1f}  ({r['baseline_top5']:>4.0f}% -> {r['new_top5']:>4.0f}%)")

    print(f"\n=== MOST IMPROVED (top {args.top_improved}) ===")
    print(f"{'class':<35} {'base':>5} -> {'new':>5}  d {'top1':>5}")
    print("-" * 70)
    for r in rows_by_imp[:args.top_improved]:
        if r["delta_top1"] <= 0: break
        print(f"{r['class']:<35} {r['baseline_top1']:>4.0f}% -> {r['new_top1']:>4.0f}%  "
              f"d {r['delta_top1']:>+5.1f}")

    print(f"\n=== MOST REGRESSED (top {args.top_regressed}) ===")
    print(f"{'class':<35} {'base':>5} -> {'new':>5}  d {'top1':>5}")
    print("-" * 70)
    for r in rows_by_reg[:args.top_regressed]:
        if r["delta_top1"] >= 0: break
        print(f"{r['class']:<35} {r['baseline_top1']:>4.0f}% -> {r['new_top1']:>4.0f}%  "
              f"d {r['delta_top1']:>+5.1f}")

    rows.sort(key=lambda r: -r["delta_top1"])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCombined CSV written: {args.out}")

if __name__ == "__main__":
    main()
