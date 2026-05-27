import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

def collect_probs(mod_name: str, ckpt_path: str, device: torch.device,
                  class_to_idx: dict, is_v6: bool):
    mod = importlib.import_module(mod_name)
    ds = mod.HAA500Dataset("test", class_to_idx=class_to_idx, augment=False)

    if is_v6:
        model = mod.HierarchicalSTGCN(num_classes=500).to(device)
    else:
        model = mod.ThreeStreamSTGCN(num_classes=500).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers=False)

    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            if is_v6:
                body, lhand, rhand, face, y = batch
                body  = body.to(device,  non_blocking=True)
                lhand = lhand.to(device, non_blocking=True)
                rhand = rhand.to(device, non_blocking=True)
                face  = face.to(device,  non_blocking=True)
                y     = y.to(device,     non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(body, lhand, rhand, face)
                    b2, l2, r2, f2 = mod._hflip_parts(body, lhand, rhand, face)
                    logits2 = model(b2, l2, r2, f2)
                    probs   = (logits.softmax(-1) + logits2.softmax(-1)) / 2
            else:
                joint, bone, motion, y = batch
                joint  = joint.to(device,  non_blocking=True)
                bone   = bone.to(device,   non_blocking=True)
                motion = motion.to(device, non_blocking=True)
                y      = y.to(device,      non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(joint, bone, motion)
                    j2, b2, m2 = mod._hflip_streams(joint, bone, motion)
                    logits2 = model(j2, b2, m2)
                    probs   = (logits.softmax(-1) + logits2.softmax(-1)) / 2
            all_probs.append(probs.float().cpu().numpy())
            all_labels.append(y.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

def top1(probs: np.ndarray, labels: np.ndarray) -> float:
    return 100.0 * (probs.argmax(1) == labels).mean()

def top5(probs: np.ndarray, labels: np.ndarray) -> float:
    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    hit = (top5_idx == labels[:, None]).any(axis=1)
    return 100.0 * hit.mean()

def load_per_class_accs(path: Path) -> dict:
    acc = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            acc[row["class"]] = float(row["top1_acc"])
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-checkpoint", default=str(ROOT / "best_stgcn_v4_emattta.pth"))
    ap.add_argument("--v6-checkpoint", default=str(ROOT / "best_stgcn_v6_combined.pth"))
    ap.add_argument("--v4-csv",        default=str(ROOT / "per_class_accuracy_v4_test.csv"))
    ap.add_argument("--v6-csv",        default=str(ROOT / "per_class_accuracy_v6_test.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    import st_gcn_v4 as v4
    train_ds = v4.HAA500Dataset("train", augment=False)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    print(f"Classes: {len(class_to_idx)}")

    print("Running v4 on test (with TTA) ...")
    v4_probs, labels = collect_probs("st_gcn_v4", args.v4_checkpoint, device, class_to_idx, is_v6=False)
    print(f"  v4 done. Shape: {v4_probs.shape}")

    print("Running v6 on test (with TTA) ...")
    v6_probs, labels6 = collect_probs("st_gcn_v6", args.v6_checkpoint, device, class_to_idx, is_v6=True)
    print(f"  v6 done. Shape: {v6_probs.shape}")

    assert np.array_equal(labels, labels6), "v4 and v6 datasets must traverse classes in the same order"

    v4_per_class = load_per_class_accs(Path(args.v4_csv))
    v6_per_class = load_per_class_accs(Path(args.v6_csv))

    w_v4 = np.zeros(500, dtype=np.float32)
    w_v6 = np.zeros(500, dtype=np.float32)
    for c, idx in class_to_idx.items():
        a4 = v4_per_class.get(c, 0.0)
        a6 = v6_per_class.get(c, 0.0)
        if a4 + a6 < 1e-6:
            w_v4[idx] = w_v6[idx] = 0.5
        else:
            w_v4[idx] = a4 / (a4 + a6)
            w_v6[idx] = a6 / (a4 + a6)

    avg_probs = 0.5 * v4_probs + 0.5 * v6_probs
    trust_probs = w_v4[None, :] * v4_probs + w_v6[None, :] * v6_probs
    trust_probs = trust_probs / (trust_probs.sum(axis=1, keepdims=True) + 1e-9)

    v4_conf = v4_probs.max(axis=1)
    v6_conf = v6_probs.max(axis=1)
    pick_v6 = v6_conf > v4_conf
    conf_probs = np.where(pick_v6[:, None], v6_probs, v4_probs)

    oracle_predictions = np.empty(len(labels), dtype=np.int64)
    for i, true_c in enumerate(labels):
        true_class_name = idx_to_class[int(true_c)]
        a4 = v4_per_class.get(true_class_name, 0.0)
        a6 = v6_per_class.get(true_class_name, 0.0)
        if a4 > a6:
            oracle_predictions[i] = v4_probs[i].argmax()
        elif a6 > a4:
            oracle_predictions[i] = v6_probs[i].argmax()
        else:
            oracle_predictions[i] = avg_probs[i].argmax()
    oracle_top1 = 100.0 * (oracle_predictions == labels).mean()

    rule_predictions = np.empty(len(labels), dtype=np.int64)
    for i in range(len(labels)):
        p4 = v4_probs[i].argmax()
        p6 = v6_probs[i].argmax()
        cn4 = idx_to_class[int(p4)]
        cn6 = idx_to_class[int(p6)]
        trust4_for_p4 = v4_per_class.get(cn4, 0.0) >= v6_per_class.get(cn4, 0.0)
        trust6_for_p6 = v6_per_class.get(cn6, 0.0) >= v4_per_class.get(cn6, 0.0)
        if trust4_for_p4 and not trust6_for_p6:
            rule_predictions[i] = p4
        elif trust6_for_p6 and not trust4_for_p4:
            rule_predictions[i] = p6
        else:
            rule_predictions[i] = avg_probs[i].argmax()
    rule_top1 = 100.0 * (rule_predictions == labels).mean()

    print("\n=== RESULTS on test split (1500 samples) ===")
    print(f"{'Strategy':<55} {'Top-1':>7} {'Top-5':>7}  Notes")
    print("-" * 100)
    print(f"{'v4 alone (TTA)':<55} {top1(v4_probs, labels):>6.2f}% {top5(v4_probs, labels):>6.2f}%  baseline body-only")
    print(f"{'v6 alone (TTA)':<55} {top1(v6_probs, labels):>6.2f}% {top5(v6_probs, labels):>6.2f}%  baseline hierarchical")
    print()
    print(f"{'Deployable strategies (no oracle):':<55}")
    print(f"{'  Simple average  0.5*v4 + 0.5*v6':<55} {top1(avg_probs, labels):>6.2f}% {top5(avg_probs, labels):>6.2f}%  no class info needed")
    print(f"{'  Confidence-pick  per-sample max-prob model':<55} {top1(conf_probs, labels):>6.2f}% {top5(conf_probs, labels):>6.2f}%  no class info needed")
    print()
    print(f"{'Oracle / upper-bound strategies (use test-derived class accs):':<55}")
    print(f"{'  Per-class probability blend':<55} {top1(trust_probs, labels):>6.2f}% {top5(trust_probs, labels):>6.2f}%  trust_w[c] = acc_v4[c] / (acc_v4[c]+acc_v6[c])")
    print(f"{'  Per-class router (true-class oracle)':<55} {oracle_top1:>6.2f}% {'-':>7}  picks better model given true class")
    print(f"{'  Per-class router (predicted-class voting)':<55} {rule_top1:>6.2f}% {'-':>7}  routes based on each model's prediction class")

    out_csv = ROOT / "ensemble_v4_v6_per_class.csv"
    n = len(labels)
    per_class_stats = {idx_to_class[i]: {"n": 0, "v4_hit": 0, "v6_hit": 0, "avg_hit": 0,
                                          "conf_hit": 0, "trust_hit": 0, "oracle_hit": 0}
                       for i in range(500)}
    v4_pred  = v4_probs.argmax(axis=1)
    v6_pred  = v6_probs.argmax(axis=1)
    avg_pred = avg_probs.argmax(axis=1)
    conf_pred  = conf_probs.argmax(axis=1)
    trust_pred = trust_probs.argmax(axis=1)
    for i in range(n):
        cname = idx_to_class[int(labels[i])]
        s = per_class_stats[cname]
        s["n"] += 1
        s["v4_hit"]    += int(v4_pred[i]    == labels[i])
        s["v6_hit"]    += int(v6_pred[i]    == labels[i])
        s["avg_hit"]   += int(avg_pred[i]   == labels[i])
        s["conf_hit"]  += int(conf_pred[i]  == labels[i])
        s["trust_hit"] += int(trust_pred[i] == labels[i])
        s["oracle_hit"]+= int(oracle_predictions[i] == labels[i])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "samples",
                    "v4_top1", "v6_top1", "avg_top1", "conf_top1", "trust_top1", "oracle_top1"])
        for cname, s in sorted(per_class_stats.items()):
            if s["n"] == 0:
                continue
            n_ = s["n"]
            w.writerow([cname, n_,
                        100*s["v4_hit"]/n_, 100*s["v6_hit"]/n_,
                        100*s["avg_hit"]/n_, 100*s["conf_hit"]/n_,
                        100*s["trust_hit"]/n_, 100*s["oracle_hit"]/n_])
    print(f"\nPer-class detail written: {out_csv}")

if __name__ == "__main__":
    main()
