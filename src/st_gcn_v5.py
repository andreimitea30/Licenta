"""
ST-GCN v5 — extends v4 (3-stream ST-GCN) from a 33-node body skeleton to a
75-node body+hands skeleton.

Skeleton layout (NUM_NODES=75):
  Indices  0..32  : 33 body landmarks (MediaPipe Pose convention)
  Indices 33..53  : 21 LEFT hand landmarks (MediaPipe Hand convention; wrist at 33)
  Indices 54..74  : 21 RIGHT hand landmarks (MediaPipe Hand convention; wrist at 54)

Reads from extracted_skeletons_holistic/{train|val|test}/<class>/<video>.npy
produced by src/pose_extraction_holistic.py. No 33-node fallback because the
two layouts are not interchangeable.

Everything else is inherited from v4: 3-stream joint+bone+motion fusion,
DROPOUT=0.5, MIXUP_ALPHA=0.3, attention on blocks [8,9], EMA + TTA enabled
by default, clip-level augmentation, plain shuffled DataLoader.
"""
import copy
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT_DIR        = Path(__file__).resolve().parent.parent
FEATURES_DIR    = ROOT_DIR / "extracted_skeletons_holistic"
MODEL_SAVE_PATH = ROOT_DIR / "best_stgcn_v5.pth"
CHECKPOINT_PATH = ROOT_DIR / "checkpoint_v5.pth"

NUM_EPOCHS    = 150
BATCH_SIZE    = 16   # reduced from 32 to fit V=75 in 8GB without memory pressure
BASE_LR       = 0.001
WEIGHT_DECAY  = 5e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTH  = 0.15
GRAD_CLIP     = 1.0
MAX_FRAMES    = 60
IN_CHANNELS   = 4   # x/y/z/vis OR vx/vy/vz/vis OR bx/by/bz/vis
DROPOUT       = 0.5
DROP_PATH_MAX = 0.3
MIXUP_ALPHA   = 0.3

# Free-win recipes (ablation toggles)
USE_EMA   = True
USE_TTA   = True
EMA_DECAY = 0.999

# Skeleton topology
NUM_BODY        = 33
NUM_HAND        = 21
LEFT_HAND_BASE  = NUM_BODY                 # 33
RIGHT_HAND_BASE = NUM_BODY + NUM_HAND      # 54
NUM_NODES       = NUM_BODY + 2 * NUM_HAND  # 75

# Body left/right pairs (from MediaPipe pose), unchanged from v4.
_BODY_FLIP_PAIRS = [
    (1,4),(2,5),(3,6),(7,8),(9,10),
    (11,12),(13,14),(15,16),(17,18),(19,20),(21,22),
    (23,24),(25,26),(27,28),(29,30),(31,32),
]
# Hand left/right mirror: i-th left-hand landmark <-> i-th right-hand landmark.
_HAND_FLIP_PAIRS = [(LEFT_HAND_BASE + i, RIGHT_HAND_BASE + i)
                    for i in range(NUM_HAND)]
FLIP_PAIRS = _BODY_FLIP_PAIRS + _HAND_FLIP_PAIRS

# Body edges (MediaPipe pose graph), unchanged from v4.
_BODY_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]
# MediaPipe HandLandmarker topology (intra-hand connections).
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17),                                # Palm closure
]
# Cross-part connectors: body wrist -> hand wrist
_BODY_LEFT_WRIST  = 15
_BODY_RIGHT_WRIST = 16
_CONNECTORS = [
    (_BODY_LEFT_WRIST,  LEFT_HAND_BASE),
    (_BODY_RIGHT_WRIST, RIGHT_HAND_BASE),
]
EDGES = (
    list(_BODY_EDGES)
    + [(LEFT_HAND_BASE + i, LEFT_HAND_BASE + j)   for (i, j) in _HAND_EDGES]
    + [(RIGHT_HAND_BASE + i, RIGHT_HAND_BASE + j) for (i, j) in _HAND_EDGES]
    + _CONNECTORS
)


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


def augment_skeleton(data: np.ndarray) -> np.ndarray:
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


class HAA500Dataset(Dataset):
    def __init__(self, split, class_to_idx=None, max_frames=MAX_FRAMES, augment=False):
        self.max_frames = max_frames
        self.augment    = augment

        root = FEATURES_DIR / split
        self.file_paths = sorted(root.rglob("*.npy"))

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

        T, V, C = data.shape
        C_pos = 3 if C == 4 else 2
        pos    = data[:, :, :C_pos]                # (T, V, C_pos)
        vis_ch = data[:, :, C_pos:C_pos + 1]       # (T, V, 1)

        # Bone vectors: edge-wise positional differences.
        bone_pos = compute_bone_features(data)     # (T, V, C_pos)

        # Joint velocities: temporal first differences (frame 0 stays zero).
        velocity = np.zeros_like(pos)
        velocity[1:] = pos[1:] - pos[:-1]

        def pack_4ch(features_3d_or_2d: np.ndarray) -> np.ndarray:
            """Pack (T, V, C_pos) features + visibility into IN_CHANNELS=4 channels.
            For 2D inputs (C_pos=2), the z slot is zero-padded so the model sees a
            consistent 4-channel layout regardless of source modality."""
            if C_pos == 3:
                return np.concatenate([features_3d_or_2d, vis_ch], axis=2)
            zero_z = np.zeros_like(vis_ch)
            return np.concatenate([features_3d_or_2d, zero_z, vis_ch], axis=2)

        joint  = pack_4ch(pos)
        bone   = pack_4ch(bone_pos)
        motion = pack_4ch(velocity)

        def to_model_tensor(x):
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, axis=-1)
            return torch.from_numpy(x)

        return to_model_tensor(joint), to_model_tensor(bone), to_model_tensor(motion)

    def __getitem__(self, idx):
        path       = self.file_paths[idx]
        class_name = path.parent.name
        label      = self.class_to_idx[class_name]
        data       = np.load(path).astype(np.float32)
        joint, bone, motion = self._to_tensor(data)
        return joint, bone, motion, torch.tensor(label, dtype=torch.long)


class Graph:
    def __init__(self, num_node=NUM_NODES):
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
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_NODES)
        self.blocks  = nn.ModuleList([
            STGCNBlock(in_channels, 64,  A, residual=False, dropout=dropout, drop_path_rate=dpr[0]),
            STGCNBlock(64,  64,  A, dropout=dropout, drop_path_rate=dpr[1]),
            STGCNBlock(64,  64,  A, dropout=dropout, drop_path_rate=dpr[2]),
            STGCNBlock(64,  64,  A, dropout=dropout, drop_path_rate=dpr[3]),
            STGCNBlock(64,  128, A, stride=2, dropout=dropout, drop_path_rate=dpr[4]),
            STGCNBlock(128, 128, A, dropout=dropout, drop_path_rate=dpr[5]),
            STGCNBlock(128, 128, A, dropout=dropout, drop_path_rate=dpr[6]),
            STGCNBlock(128, 256, A, stride=2, dropout=dropout, drop_path_rate=dpr[7]),
            STGCNBlock(256, 256, A, dropout=dropout, drop_path_rate=dpr[8], use_attn=True),
            STGCNBlock(256, 256, A, dropout=dropout, drop_path_rate=dpr[9], use_attn=True),
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


class ThreeStreamSTGCN(nn.Module):
    def __init__(self, num_classes=500):
        super().__init__()
        self.joint_stream  = SingleStream(num_classes)
        self.bone_stream   = SingleStream(num_classes)
        self.motion_stream = SingleStream(num_classes)

    def forward(self, joint, bone, motion):
        return (self.joint_stream(joint)
                + self.bone_stream(bone)
                + self.motion_stream(motion)) / 3


def mixup(joint, bone, motion, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(joint.size(0), device=joint.device)
    return (lam*joint  + (1-lam)*joint[idx],
            lam*bone   + (1-lam)*bone[idx],
            lam*motion + (1-lam)*motion[idx],
            y, y[idx], lam)

def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1-lam) * crit(pred, yb)


class ModelEMA:
    """Exponential moving average of model parameters and buffers.
    Float-tensor entries get EMA-updated; integer buffers (e.g. BN's
    num_batches_tracked) are copied from the live model."""
    def __init__(self, model, decay=EMA_DECAY):
        self.decay  = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ev, mv in zip(self.module.state_dict().values(),
                          model.state_dict().values()):
            if ev.dtype.is_floating_point:
                ev.mul_(self.decay).add_(mv.detach(), alpha=1.0 - self.decay)
            else:
                ev.copy_(mv)

    def state_dict(self):
        return {"decay": self.decay, "module": self.module.state_dict()}

    def load_state_dict(self, sd):
        self.decay = sd.get("decay", self.decay)
        self.module.load_state_dict(sd["module"])


def _hflip_streams(joint, bone, motion):
    """Tensor-level horizontal flip on already-packed (N, C, T, V, M) inputs.
    Negates the x-component (channel 0) of each stream and swaps paired joints
    in the V dimension. For joint stream this is exact; for bone/motion it is
    the natural mirror operation given how they're derived from positions."""
    def _flip(x):
        x = x.clone()
        x[:, 0] *= -1
        for l, r in FLIP_PAIRS:
            x[:, :, :, [l, r], :] = x[:, :, :, [r, l], :]
        return x
    return _flip(joint), _flip(bone), _flip(motion)


def _evaluate(model, loader, device, use_amp, criterion, use_tta: bool):
    model.eval()
    loss_sum = top1 = top5 = total = 0
    with torch.no_grad():
        for joint, bone, motion, y in loader:
            joint  = joint.to(device,  non_blocking=True)
            bone   = bone.to(device,   non_blocking=True)
            motion = motion.to(device, non_blocking=True)
            y      = y.to(device,      non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(joint, bone, motion)
                if use_tta:
                    j2, b2, m2 = _hflip_streams(joint, bone, motion)
                    logits2 = model(j2, b2, m2)
                    probs = (logits.softmax(-1) + logits2.softmax(-1)) / 2
                    loss  = criterion(probs.log(), y)  # NLL on averaged probs ~ XE
                else:
                    probs = logits.softmax(-1)
                    loss  = criterion(logits, y)
            loss_sum += loss.item()
            top1 += probs.argmax(1).eq(y).sum().item()
            top5 += probs.topk(5, 1)[1].eq(y.view(-1, 1)).any(1).sum().item()
            total += y.size(0)
    return loss_sum / max(1, len(loader)), 100.0 * top1 / total, 100.0 * top5 / total


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
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset([train_split_a, train_split_b])
        class_to_idx = train_split_a.class_to_idx
        val_ds = HAA500Dataset("test", class_to_idx=class_to_idx, augment=False)
        print("Combined train+val mode — monitoring on test split")
    else:
        train_ds     = HAA500Dataset("train", augment=True)
        class_to_idx = train_ds.class_to_idx
        val_ds       = HAA500Dataset("val", class_to_idx=class_to_idx, augment=False)

    print(f"Train: {len(train_ds)}  Val/Test: {len(val_ds)}")

    kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=use_amp, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    model     = ThreeStreamSTGCN(num_classes=500).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    warmup    = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, total_iters=WARMUP_EPOCHS)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs-WARMUP_EPOCHS, eta_min=BASE_LR*0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema       = ModelEMA(model) if USE_EMA else None
    print(f"USE_EMA={USE_EMA}  USE_TTA={USE_TTA}  EMA_DECAY={EMA_DECAY}")

    start_epoch = 0; best_acc = 0.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        if ema is not None and 'ema' in ck:
            ema.load_state_dict(ck['ema'])
        start_epoch = ck['epoch'] + 1; best_acc = ck['best_acc']
        print(f"Resumed epoch {start_epoch}  best={best_acc:.1f}%")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for joint, bone, motion, y in tqdm(train_loader, desc=f"E{epoch+1:03d}/{num_epochs} train", leave=False):
            joint  = joint.to(device,  non_blocking=True)
            bone   = bone.to(device,   non_blocking=True)
            motion = motion.to(device, non_blocking=True)
            y      = y.to(device,      non_blocking=True)
            joint_m, bone_m, motion_m, ya, yb, lam = mixup(joint, bone, motion, y)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model(joint_m, bone_m, motion_m)
                loss = mixup_loss(criterion, out, ya, yb, lam)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            if ema is not None:
                ema.update(model)
            tr_loss    += loss.item()
            pred_y = ya if lam >= 0.5 else yb
            tr_correct += out.argmax(1).eq(pred_y).sum().item()
            tr_total   += y.size(0)
        scheduler.step()

        # Live model evaluation (no TTA) — keeps the legacy log line so prior
        # parsers and ablation scripts continue to work.
        live_loss, live_top1, live_top5 = _evaluate(
            model, val_loader, device, use_amp, criterion, use_tta=False)

        live_tta_top1 = live_tta_top5 = None
        ema_top1     = ema_top5     = None
        ema_tta_top1 = ema_tta_top5 = None
        if USE_TTA:
            _, live_tta_top1, live_tta_top5 = _evaluate(
                model, val_loader, device, use_amp, criterion, use_tta=True)
        if ema is not None:
            _, ema_top1, ema_top5 = _evaluate(
                ema.module, val_loader, device, use_amp, criterion, use_tta=False)
            if USE_TTA:
                _, ema_tta_top1, ema_tta_top5 = _evaluate(
                    ema.module, val_loader, device, use_amp, criterion, use_tta=True)

        tr_acc  = 100.0 * tr_correct / tr_total
        lr_now  = optimizer.param_groups[0]['lr']
        # Legacy line: matches v2/v4-baseline format so old regex still parses.
        print(f"Epoch {epoch+1:3d}/{num_epochs} | LR {lr_now:.2e} | "
              f"Train {tr_loss/len(train_loader):.4f}/{tr_acc:.1f}% | "
              f"Val Top-1 {live_top1:.1f}% Top-5 {live_top5:.1f}%")
        # Extra ablation lines.
        if live_tta_top1 is not None:
            print(f"  Live+TTA   Top-1 {live_tta_top1:.1f}% Top-5 {live_tta_top5:.1f}%")
        if ema_top1 is not None:
            print(f"  EMA        Top-1 {ema_top1:.1f}% Top-5 {ema_top5:.1f}%")
        if ema_tta_top1 is not None:
            print(f"  EMA+TTA    Top-1 {ema_tta_top1:.1f}% Top-5 {ema_tta_top5:.1f}%")

        # Best-checkpoint metric: prefer EMA+TTA when available, else fall through.
        score = ema_tta_top1 if ema_tta_top1 is not None \
                else (ema_top1 if ema_top1 is not None
                      else (live_tta_top1 if live_tta_top1 is not None else live_top1))

        ckpt_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc': best_acc,
        }
        if ema is not None:
            ckpt_state['ema'] = ema.state_dict()
        torch.save(ckpt_state, ckpt_path)

        if score > best_acc:
            best_acc = score
            # Save the EMA module if we have it (it's the eval-time weights);
            # otherwise save the live model.
            torch.save((ema.module if ema is not None else model).state_dict(), model_path)
            print(f"  --> New best (selection metric): {best_acc:.1f}%")


if __name__ == "__main__":
    train()
