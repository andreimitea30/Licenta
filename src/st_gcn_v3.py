
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from tqdm import tqdm

ROOT_DIR        = Path(__file__).resolve().parent.parent
FEATURES_DIR    = ROOT_DIR / "extracted_skeletons_world"
FALLBACK_DIR    = ROOT_DIR / "extracted_skeletons"
MODEL_SAVE_PATH = ROOT_DIR / "best_stgcn_v3.pth"
CHECKPOINT_PATH = ROOT_DIR / "checkpoint_v3.pth"

NUM_EPOCHS    = 150
BATCH_SIZE    = 32
BASE_LR       = 0.001
WEIGHT_DECAY  = 5e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTH  = 0.15
GRAD_CLIP     = 1.0
MAX_FRAMES    = 60
IN_CHANNELS   = 7
DROPOUT       = 0.5
DROP_PATH_MAX = 0.3
MIXUP_ALPHA   = 0.3

TEMPORAL_CUTOUT_PROB = 0.3
JOINT_DROPOUT_PROB   = 0.3

ATTENTION_BLOCKS      = [3, 6, 9]
USE_WEIGHTED_SAMPLER  = True

FLIP_PAIRS = [
    (1,4),(2,5),(3,6),(7,8),(9,10),
    (11,12),(13,14),(15,16),(17,18),(19,20),(21,22),
    (23,24),(25,26),(27,28),(29,30),(31,32),
]

EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

def normalize_skeleton(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    C = data.shape[2]
    if C == 4:
        scale = np.linalg.norm(data[:, 11, :3] - data[:, 12, :3], axis=1).mean()
        if scale > 1e-4:
            data[:, :, :3] /= scale
    else:
        hip = (data[:, 23, :2] + data[:, 24, :2]) / 2
        data[:, :, :2] -= hip[:, np.newaxis, :]
        scale = np.linalg.norm(data[:, 11, :2] - data[:, 12, :2], axis=1).mean()
        if scale > 1e-4:
            data[:, :, :2] /= scale
    return data

def compute_bone_features(data: np.ndarray) -> np.ndarray:
    T, V, C = data.shape
    C_pos = 3 if C == 4 else 2
    bone = np.zeros((T, V, C_pos), dtype=np.float32)
    cnt  = np.zeros(V, dtype=np.float32)
    for (i, j) in EDGES:
        bone[:, i, :] += data[:, j, :C_pos] - data[:, i, :C_pos]
        bone[:, j, :] += data[:, i, :C_pos] - data[:, j, :C_pos]
        cnt[i] += 1; cnt[j] += 1
    cnt = np.maximum(cnt, 1)
    bone /= cnt[np.newaxis, :, np.newaxis]
    return bone

def add_motion_features(data: np.ndarray) -> np.ndarray:
    C = data.shape[2]
    C_pos = 3 if C == 4 else 2
    vel = np.zeros_like(data[:, :, :C_pos])
    vel[1:] = data[1:, :, :C_pos] - data[:-1, :, :C_pos]
    return np.concatenate([data, vel], axis=2)

def temporal_sample(data: np.ndarray, n: int, augment: bool) -> np.ndarray:
    T = data.shape[0]
    if T >= n:
        start = np.random.randint(0, T - n + 1) if augment else 0
        if not augment:
            idx = np.linspace(0, T-1, n, dtype=int)
            return data[idx]
        return data[start:start+n]
    pad = np.repeat(data[-1:], n - T, axis=0)
    return np.concatenate([data, pad], axis=0)

def _augment_v2(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    C = data.shape[2]
    C_pos = 3 if C >= 4 else 2
    if np.random.random() < 0.5:
        data[:, :, 0] *= -1
        for l, r in FLIP_PAIRS:
            data[:, [l, r]] = data[:, [r, l]]
    if np.random.random() < 0.2:
        data = data[::-1].copy()
    data[:, :, :C_pos] *= np.random.uniform(0.85, 1.15)
    noise = np.random.randn(data.shape[0], data.shape[1], C_pos).astype(np.float32) * 0.02
    data[:, :, :C_pos] += noise
    return data

def _augment_v3(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    T, V, C = data.shape
    C_pos = 3 if C >= 4 else 2

    if np.random.random() < 0.5:
        data[:, :, 0] *= -1
        for l, r in FLIP_PAIRS:
            data[:, [l, r]] = data[:, [r, l]]

    if np.random.random() < 0.2:
        data = data[::-1].copy()

    scales = np.random.uniform(0.9, 1.1, size=(T, 1, 1)).astype(np.float32)
    data[:, :, :C_pos] *= scales

    if C_pos == 2:
        theta = np.random.normal(0.0, np.deg2rad(5.0), size=T).astype(np.float32)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.stack([np.stack([cos_t, -sin_t], axis=-1),
                      np.stack([sin_t,  cos_t], axis=-1)], axis=-2)
        data[:, :, :2] = np.einsum('tij,tvj->tvi', R, data[:, :, :2])

    noise = np.random.randn(T, V, C_pos).astype(np.float32) * 0.02
    data[:, :, :C_pos] += noise

    if np.random.random() < TEMPORAL_CUTOUT_PROB and T > 2:
        length = int(np.random.randint(2, min(9, T)))
        start  = int(np.random.randint(0, T - length + 1))
        data[start:start + length] = 0.0

    if np.random.random() < JOINT_DROPOUT_PROB:
        k = int(np.random.randint(1, 4))
        idx = np.random.choice(V, size=k, replace=False)
        data[:, idx, :] = 0.0

    return data

def augment_skeleton(data: np.ndarray) -> np.ndarray:
    return _augment_v2(data) if os.environ.get('V3_AUGMENT_MODE', 'v3') == 'v2' else _augment_v3(data)

class HAA500Dataset(Dataset):
    def __init__(self, split, class_to_idx=None, max_frames=MAX_FRAMES, augment=False):
        self.max_frames = max_frames
        self.augment    = augment

        primary  = FEATURES_DIR / split
        fallback = FALLBACK_DIR / split

        self.file_paths = []
        for path in sorted(fallback.rglob("*.npy")):
            world_path = primary / path.relative_to(fallback)
            self.file_paths.append(world_path if world_path.exists() else path)

        if class_to_idx is None:
            folders = sorted({p.parent.name for p in self.file_paths})
            self.class_to_idx = {c: i for i, c in enumerate(folders)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_paths)

    def _to_tensor(self, data):
        data = normalize_skeleton(data)
        data = temporal_sample(data, self.max_frames, self.augment)
        if self.augment:
            data = augment_skeleton(data)

        joint = add_motion_features(data)
        if joint.shape[2] < IN_CHANNELS:
            joint = np.concatenate(
                [joint, np.zeros((joint.shape[0], joint.shape[1],
                                  IN_CHANNELS - joint.shape[2]), dtype=np.float32)], axis=2)
        joint = joint[:, :, :IN_CHANNELS]

        C = data.shape[2]
        C_pos = 3 if C == 4 else 2
        bone_pos = compute_bone_features(data)
        bone_vel = np.zeros_like(bone_pos)
        bone_vel[1:] = bone_pos[1:] - bone_pos[:-1]
        vis_ch = data[:, :, C_pos:C_pos+1]
        bone = np.concatenate([bone_pos, vis_ch, bone_vel], axis=2)
        if bone.shape[2] < IN_CHANNELS:
            bone = np.concatenate(
                [bone, np.zeros((bone.shape[0], bone.shape[1],
                                 IN_CHANNELS - bone.shape[2]), dtype=np.float32)], axis=2)
        bone = bone[:, :, :IN_CHANNELS]

        def to_model_tensor(x):
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, axis=-1)
            return torch.from_numpy(x)

        return to_model_tensor(joint), to_model_tensor(bone)

    def __getitem__(self, idx):
        path       = self.file_paths[idx]
        class_name = path.parent.name
        label      = self.class_to_idx[class_name]
        data       = np.load(path).astype(np.float32)
        joint, bone = self._to_tensor(data)
        return joint, bone, torch.tensor(label, dtype=torch.long)

class Graph:
    def __init__(self, num_node=33):
        self.num_node = num_node
        self.A = self._build()

    def _build(self):
        A = np.zeros((self.num_node, self.num_node), dtype=np.float32)
        for i, j in EDGES:
            A[i, j] = A[j, i] = 1
        A += np.eye(self.num_node)
        d = A.sum(1) ** -0.5
        A = np.diag(d) @ A @ np.diag(d)
        return torch.tensor(A, dtype=torch.float32).unsqueeze(0)

class GraphConv(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer('A_fixed', A[0])
        self.A_learn = nn.Parameter(torch.zeros(A.shape[1], A.shape[2]))
        self.conv    = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return torch.einsum('nctv,vw->nctw', self.conv(x), self.A_fixed + self.A_learn)

class MultiScaleTCN(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        assert channels % 4 == 0
        mid = channels // 4
        def _branch(k):
            return nn.Sequential(
                nn.Conv2d(channels, mid, (k, 1), stride=(stride, 1), padding=(k//2, 0)),
                nn.BatchNorm2d(mid), nn.ReLU(),
            )
        self.b3    = _branch(3)
        self.b7    = _branch(7)
        self.b9    = _branch(9)
        self.bpool = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.Conv2d(channels, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(),
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(torch.cat([self.b3(x), self.b7(x), self.b9(x), self.bpool(x)], dim=1))

class TemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        N, C, T, V = x.shape
        orig_dtype = x.dtype
        x_f32 = x.float()
        q = x_f32.permute(0, 3, 2, 1).reshape(N * V, T, C)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            out, _ = self.attn(q, q, q)
        out = self.norm(q + out).reshape(N, V, T, C).permute(0, 3, 2, 1)
        return x + out.to(orig_dtype)

class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, residual=True,
                 dropout=DROPOUT, drop_path_rate=0.0, use_attn=False):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.gcn  = GraphConv(in_ch, out_ch, A)
        self.bn1  = nn.BatchNorm2d(out_ch)
        self.tcn  = MultiScaleTCN(out_ch, stride=stride)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=False)
        self.attn = TemporalAttention(out_ch) if use_attn else None

        if not residual:
            self.residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.training and self.drop_path_rate > 0 and torch.rand(1).item() < self.drop_path_rate:
            return self.relu(res) if not isinstance(res, int) else x
        out = self.drop(self.tcn(self.relu(self.bn1(self.gcn(x)))))
        out = self.relu(out + res)
        if self.attn is not None:
            out = self.attn(out)
        return out

class SingleStream(nn.Module):
    def __init__(self, num_classes=500, in_channels=IN_CHANNELS,
                 dropout=DROPOUT, drop_path_max=DROP_PATH_MAX):
        super().__init__()
        A   = Graph().A
        n   = 10
        dpr = [drop_path_max * i / (n - 1) for i in range(n)]
        self.data_bn = nn.BatchNorm1d(in_channels * 33)
        attn_set = set(ATTENTION_BLOCKS)
        block_specs = [
            (in_channels, 64,  1, False),
            (64,  64,  1, True),
            (64,  64,  1, True),
            (64,  64,  1, True),
            (64,  128, 2, True),
            (128, 128, 1, True),
            (128, 128, 1, True),
            (128, 256, 2, True),
            (256, 256, 1, True),
            (256, 256, 1, True),
        ]
        self.blocks = nn.ModuleList([
            STGCNBlock(ic, oc, A, stride=st, residual=res,
                       dropout=dropout, drop_path_rate=dpr[i],
                       use_attn=(i in attn_set))
            for i, (ic, oc, st, res) in enumerate(block_specs)
        ])
        self.fc = nn.Linear(256, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for blk in self.blocks:
            x = blk(x)
        x = F.adaptive_avg_pool2d(x, 1).view(N, M, -1).mean(dim=1)
        return self.fc(x)

class TwoStreamSTGCN(nn.Module):
    def __init__(self, num_classes=500):
        super().__init__()
        self.joint_stream = SingleStream(num_classes)
        self.bone_stream  = SingleStream(num_classes)

    def forward(self, joint, bone):
        return (self.joint_stream(joint) + self.bone_stream(bone)) / 2

def mixup(joint, bone, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(joint.size(0), device=joint.device)
    return (lam*joint + (1-lam)*joint[idx],
            lam*bone  + (1-lam)*bone[idx],
            y, y[idx], lam)

def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1-lam) * crit(pred, yb)

def train(features_dir=None, num_epochs=NUM_EPOCHS,
          checkpoint_path=None, model_save_path=None,
          combined_train=False):
    global FEATURES_DIR
    if features_dir:
        FEATURES_DIR = Path(features_dir)
    ckpt_path  = Path(checkpoint_path)  if checkpoint_path  else CHECKPOINT_PATH
    model_path = Path(model_save_path)  if model_save_path  else MODEL_SAVE_PATH

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}  AMP: {use_amp}")
    if use_amp:
        torch.backends.cudnn.benchmark = True

    if combined_train:
        train_split_a = HAA500Dataset("train", augment=True)
        train_split_b = HAA500Dataset("val", class_to_idx=train_split_a.class_to_idx, augment=True)
        train_ds = ConcatDataset([train_split_a, train_split_b])
        class_to_idx = train_split_a.class_to_idx
        val_ds = HAA500Dataset("test", class_to_idx=class_to_idx, augment=False)
        train_labels = [class_to_idx[p.parent.name] for p in train_split_a.file_paths] \
                     + [class_to_idx[p.parent.name] for p in train_split_b.file_paths]
        print("Combined train+val mode — monitoring on test split")
    else:
        train_ds     = HAA500Dataset("train", augment=True)
        class_to_idx = train_ds.class_to_idx
        val_ds       = HAA500Dataset("val", class_to_idx=class_to_idx, augment=False)
        train_labels = [class_to_idx[p.parent.name] for p in train_ds.file_paths]

    print(f"Train: {len(train_ds)}  Val/Test: {len(val_ds)}")

    kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=use_amp, persistent_workers=True)
    if USE_WEIGHTED_SAMPLER:
        counts  = np.bincount(train_labels, minlength=500)
        weights = (1.0 / np.maximum(counts[train_labels], 1)).astype(np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, **kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model     = TwoStreamSTGCN(num_classes=500).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    warmup    = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, total_iters=WARMUP_EPOCHS)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs-WARMUP_EPOCHS, eta_min=BASE_LR*0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch = 0; best_acc = 0.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        start_epoch = ck['epoch'] + 1; best_acc = ck['best_acc']
        print(f"Resumed epoch {start_epoch}  best={best_acc:.1f}%")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        use_mixup = MIXUP_ALPHA > 0
        for joint, bone, y in tqdm(train_loader, desc=f"E{epoch+1:03d}/{num_epochs} train", leave=False):
            joint = joint.to(device, non_blocking=True)
            bone  = bone.to(device,  non_blocking=True)
            y     = y.to(device,    non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_mixup:
                joint_in, bone_in, ya, yb, lam = mixup(joint, bone, y, MIXUP_ALPHA)
            else:
                joint_in, bone_in = joint, bone
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model(joint_in, bone_in)
                loss = mixup_loss(criterion, out, ya, yb, lam) if use_mixup else criterion(out, y)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            tr_loss    += loss.item()
            pred_y      = (ya if lam >= 0.5 else yb) if use_mixup else y
            tr_correct += out.argmax(1).eq(pred_y).sum().item()
            tr_total   += y.size(0)
        scheduler.step()

        model.eval()
        v_loss = v_top1 = v_top5 = v_total = 0
        with torch.no_grad():
            for joint, bone, y in tqdm(val_loader, desc=f"E{epoch+1:03d}/{num_epochs} val  ", leave=False):
                joint = joint.to(device, non_blocking=True)
                bone  = bone.to(device,  non_blocking=True)
                y     = y.to(device,    non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out  = model(joint, bone)
                    loss = criterion(out, y)
                v_loss += loss.item()
                v_top1 += out.argmax(1).eq(y).sum().item()
                v_top5 += out.topk(5,1)[1].eq(y.view(-1,1)).any(1).sum().item()
                v_total += y.size(0)

        tr_acc = 100.0 * tr_correct / tr_total
        top1   = 100.0 * v_top1 / v_total
        top5   = 100.0 * v_top5 / v_total
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{num_epochs} | LR {lr_now:.2e} | "
              f"Train {tr_loss/len(train_loader):.4f}/{tr_acc:.1f}% | "
              f"Val Top-1 {top1:.1f}% Top-5 {top5:.1f}%")

        torch.save({'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                    'best_acc':best_acc}, ckpt_path)
        if top1 > best_acc:
            best_acc = top1
            torch.save(model.state_dict(), model_path)
            print(f"  --> New best: {best_acc:.1f}%")

if __name__ == "__main__":
    train()
