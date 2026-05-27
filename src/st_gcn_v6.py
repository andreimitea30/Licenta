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
FEATURES_DIR    = ROOT_DIR / "extracted_skeletons_hierarchical"
MODEL_SAVE_PATH = ROOT_DIR / "best_stgcn_v6.pth"
CHECKPOINT_PATH = ROOT_DIR / "checkpoint_v6.pth"

NUM_EPOCHS    = 100
BATCH_SIZE    = 32
BASE_LR       = 0.001
WEIGHT_DECAY  = 5e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTH  = 0.15
GRAD_CLIP     = 1.0
MAX_FRAMES    = 60
IN_CHANNELS   = 12
DROPOUT       = 0.5
DROP_PATH_MAX = 0.3
MIXUP_ALPHA   = 0.3

USE_EMA   = True
USE_TTA   = True
EMA_DECAY = 0.999

NUM_BODY        = 33
NUM_HAND        = 21
LEFT_HAND_BASE  = NUM_BODY
RIGHT_HAND_BASE = NUM_BODY + NUM_HAND
FACE_BASE       = NUM_BODY + 2 * NUM_HAND
NUM_FACE        = 30
NUM_NODES       = NUM_BODY + 2 * NUM_HAND + NUM_FACE

BODY_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
FACE_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),
    (8,9),(10,11),
    (12,13),(12,14),(13,15),(14,15),
    (16,17),(16,18),(17,19),(18,19),
    (20,21),
    (22,23),(22,24),(23,25),(24,25),
    (26,27),(26,28),(27,29),(28,29),
    (22,26),(23,27),
]

BODY_FLIP_PAIRS = [
    (1,4),(2,5),(3,6),(7,8),(9,10),
    (11,12),(13,14),(15,16),(17,18),(19,20),(21,22),
    (23,24),(25,26),(27,28),(29,30),(31,32),
]
HAND_INTERSWAP = [(LEFT_HAND_BASE + i, RIGHT_HAND_BASE + i)
                  for i in range(NUM_HAND)]
FACE_FLIP_PAIRS_LOCAL = [
    (2,3),(4,7),(5,6),
    (8,11),(9,10),
    (12,17),(13,16),(14,18),(15,19),
    (22,23),(26,27),
]
FACE_FLIP_PAIRS = [(FACE_BASE + a, FACE_BASE + b)
                   for (a, b) in FACE_FLIP_PAIRS_LOCAL]
FLIP_PAIRS = BODY_FLIP_PAIRS + HAND_INTERSWAP + FACE_FLIP_PAIRS

PART_SLICES = {
    "body":  (0, NUM_BODY),
    "lhand": (LEFT_HAND_BASE, LEFT_HAND_BASE + NUM_HAND),
    "rhand": (RIGHT_HAND_BASE, RIGHT_HAND_BASE + NUM_HAND),
    "face":  (FACE_BASE, FACE_BASE + NUM_FACE),
}
PART_EDGES = {
    "body":  BODY_EDGES,
    "lhand": HAND_EDGES,
    "rhand": HAND_EDGES,
    "face":  FACE_EDGES,
}
PART_NODE_COUNTS = {
    "body":  NUM_BODY,
    "lhand": NUM_HAND,
    "rhand": NUM_HAND,
    "face":  NUM_FACE,
}
PART_CHANNEL_PLAN = {
    "body":  (64, 128, 256),
    "lhand": (32, 64,  128),
    "rhand": (32, 64,  128),
    "face":  (32, 64,  128),
}
PART_NAMES = ("body", "lhand", "rhand", "face")

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

def _compute_part_bones(part_data: np.ndarray, edges) -> np.ndarray:
    T, V, C = part_data.shape
    C_pos = 3 if C == 4 else 2
    bone = np.zeros((T, V, C_pos), dtype=np.float32)
    cnt  = np.zeros(V, dtype=np.float32)
    for (i, j) in edges:
        if i >= V or j >= V:
            continue
        bone[:, i, :] += part_data[:, j, :C_pos] - part_data[:, i, :C_pos]
        bone[:, j, :] += part_data[:, i, :C_pos] - part_data[:, j, :C_pos]
        cnt[i] += 1; cnt[j] += 1
    cnt = np.maximum(cnt, 1)
    bone /= cnt[np.newaxis, :, np.newaxis]
    return bone

def temporal_sample(data: np.ndarray, n: int, augment: bool) -> np.ndarray:
    T = data.shape[0]
    if T >= n:
        if augment:
            start = np.random.randint(0, T - n + 1)
            return data[start:start+n]
        idx = np.linspace(0, T-1, n, dtype=int)
        return data[idx]
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

def pack_part_input(part_data: np.ndarray, edges) -> np.ndarray:
    T, V, C = part_data.shape
    C_pos = 3 if C == 4 else 2
    pos    = part_data[:, :, :C_pos]
    vis_ch = part_data[:, :, C_pos:C_pos+1]

    bone_pos = _compute_part_bones(part_data, edges)

    velocity = np.zeros_like(pos)
    velocity[1:] = pos[1:] - pos[:-1]

    if C_pos == 2:
        zero_z = np.zeros_like(vis_ch)
        joint  = np.concatenate([pos,      zero_z, vis_ch], axis=2)
        bone   = np.concatenate([bone_pos, zero_z, vis_ch], axis=2)
        motion = np.concatenate([velocity, zero_z, vis_ch], axis=2)
    else:
        joint  = np.concatenate([pos,      vis_ch], axis=2)
        bone   = np.concatenate([bone_pos, vis_ch], axis=2)
        motion = np.concatenate([velocity, vis_ch], axis=2)

    return np.concatenate([joint, bone, motion], axis=2).astype(np.float32)

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

    def _to_part_tensors(self, data):
        data = normalize_skeleton(data)
        data = temporal_sample(data, self.max_frames, self.augment)
        if self.augment:
            data = augment_skeleton(data)

        tensors = {}
        for name in PART_NAMES:
            a, b   = PART_SLICES[name]
            part   = data[:, a:b]
            packed = pack_part_input(part, PART_EDGES[name])
            x = np.transpose(packed, (2, 0, 1))
            x = np.expand_dims(x, axis=-1)
            tensors[name] = torch.from_numpy(x)
        return tensors

    def __getitem__(self, idx):
        path       = self.file_paths[idx]
        class_name = path.parent.name
        label      = self.class_to_idx[class_name]
        data       = np.load(path).astype(np.float32)
        parts      = self._to_part_tensors(data)
        return (parts["body"], parts["lhand"], parts["rhand"], parts["face"],
                torch.tensor(label, dtype=torch.long))

class PartGraph:
    def __init__(self, num_node, edges):
        self.num_node = num_node
        A = np.zeros((num_node, num_node), dtype=np.float32)
        for i, j in edges:
            if i < num_node and j < num_node:
                A[i, j] = A[j, i] = 1
        A += np.eye(num_node)
        d = A.sum(1) ** -0.5
        A = np.diag(d) @ A @ np.diag(d)
        self.A = torch.tensor(A, dtype=torch.float32).unsqueeze(0)

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
        n_heads = min(num_heads, max(1, channels // 16))
        while channels % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.attn = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
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

class PartSubStream(nn.Module):
    def __init__(self, num_node, edges, channel_plan,
                 in_channels=IN_CHANNELS, dropout=DROPOUT,
                 drop_path_max=DROP_PATH_MAX):
        super().__init__()
        c1, c2, c3 = channel_plan
        A = PartGraph(num_node, edges).A
        n = 10
        dpr = [drop_path_max * i / (n - 1) for i in range(n)]
        self.data_bn = nn.BatchNorm1d(in_channels * num_node)
        self.blocks  = nn.ModuleList([
            STGCNBlock(in_channels, c1, A, residual=False, dropout=dropout, drop_path_rate=dpr[0]),
            STGCNBlock(c1, c1, A, dropout=dropout, drop_path_rate=dpr[1]),
            STGCNBlock(c1, c1, A, dropout=dropout, drop_path_rate=dpr[2]),
            STGCNBlock(c1, c1, A, dropout=dropout, drop_path_rate=dpr[3]),
            STGCNBlock(c1, c2, A, stride=2, dropout=dropout, drop_path_rate=dpr[4]),
            STGCNBlock(c2, c2, A, dropout=dropout, drop_path_rate=dpr[5]),
            STGCNBlock(c2, c2, A, dropout=dropout, drop_path_rate=dpr[6]),
            STGCNBlock(c2, c3, A, stride=2, dropout=dropout, drop_path_rate=dpr[7]),
            STGCNBlock(c3, c3, A, dropout=dropout, drop_path_rate=dpr[8], use_attn=True),
            STGCNBlock(c3, c3, A, dropout=dropout, drop_path_rate=dpr[9], use_attn=True),
        ])
        self.out_dim = c3

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for blk in self.blocks:
            x = blk(x)
        x = F.adaptive_avg_pool2d(x, 1).view(N, M, -1).mean(dim=1)
        return x

class HierarchicalSTGCN(nn.Module):
    def __init__(self, num_classes=500):
        super().__init__()
        self.body  = PartSubStream(PART_NODE_COUNTS["body"],  PART_EDGES["body"],  PART_CHANNEL_PLAN["body"])
        self.lhand = PartSubStream(PART_NODE_COUNTS["lhand"], PART_EDGES["lhand"], PART_CHANNEL_PLAN["lhand"])
        self.rhand = PartSubStream(PART_NODE_COUNTS["rhand"], PART_EDGES["rhand"], PART_CHANNEL_PLAN["rhand"])
        self.face  = PartSubStream(PART_NODE_COUNTS["face"],  PART_EDGES["face"],  PART_CHANNEL_PLAN["face"])
        fused_dim  = self.body.out_dim + self.lhand.out_dim + self.rhand.out_dim + self.face.out_dim
        self.fc    = nn.Linear(fused_dim, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)

    def forward(self, body, lhand, rhand, face):
        b = self.body(body)
        l = self.lhand(lhand)
        r = self.rhand(rhand)
        f = self.face(face)
        return self.fc(torch.cat([b, l, r, f], dim=1))

def mixup(body, lhand, rhand, face, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(body.size(0), device=body.device)
    return (lam*body  + (1-lam)*body[idx],
            lam*lhand + (1-lam)*lhand[idx],
            lam*rhand + (1-lam)*rhand[idx],
            lam*face  + (1-lam)*face[idx],
            y, y[idx], lam)

def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1-lam) * crit(pred, yb)

class ModelEMA:
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

def _hflip_parts(body, lhand, rhand, face):
    def _flip_within(x, pairs):
        x = x.clone()
        x[:, 0] *= -1
        x[:, 4] *= -1
        x[:, 8] *= -1
        for l, r in pairs:
            x[:, :, :, [l, r], :] = x[:, :, :, [r, l], :]
        return x

    def _flip_handswap(x):
        x = x.clone()
        x[:, 0] *= -1
        x[:, 4] *= -1
        x[:, 8] *= -1
        return x

    body_f  = _flip_within(body, BODY_FLIP_PAIRS)
    lhand_f = _flip_handswap(rhand)
    rhand_f = _flip_handswap(lhand)
    face_f  = _flip_within(face, FACE_FLIP_PAIRS_LOCAL)
    return body_f, lhand_f, rhand_f, face_f

def _evaluate(model, loader, device, use_amp, criterion, use_tta: bool):
    model.eval()
    loss_sum = top1 = top5 = total = 0
    with torch.no_grad():
        for body, lhand, rhand, face, y in loader:
            body  = body.to(device,  non_blocking=True)
            lhand = lhand.to(device, non_blocking=True)
            rhand = rhand.to(device, non_blocking=True)
            face  = face.to(device,  non_blocking=True)
            y     = y.to(device,     non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(body, lhand, rhand, face)
                if use_tta:
                    b2, l2, r2, f2 = _hflip_parts(body, lhand, rhand, face)
                    logits2 = model(b2, l2, r2, f2)
                    probs = (logits.softmax(-1) + logits2.softmax(-1)) / 2
                    loss  = criterion(probs.log(), y)
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
        print("Combined train+val mode -- monitoring on test split")
    else:
        train_ds     = HAA500Dataset("train", augment=True)
        class_to_idx = train_ds.class_to_idx
        val_ds       = HAA500Dataset("val", class_to_idx=class_to_idx, augment=False)

    print(f"Train: {len(train_ds)}  Val/Test: {len(val_ds)}")

    kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=use_amp, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    model     = HierarchicalSTGCN(num_classes=500).to(device)
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
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

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
        for body, lhand, rhand, face, y in tqdm(train_loader, desc=f"E{epoch+1:03d}/{num_epochs} train", leave=False):
            body  = body.to(device,  non_blocking=True)
            lhand = lhand.to(device, non_blocking=True)
            rhand = rhand.to(device, non_blocking=True)
            face  = face.to(device,  non_blocking=True)
            y     = y.to(device,     non_blocking=True)
            body_m, lhand_m, rhand_m, face_m, ya, yb, lam = mixup(body, lhand, rhand, face, y)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model(body_m, lhand_m, rhand_m, face_m)
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
        print(f"Epoch {epoch+1:3d}/{num_epochs} | LR {lr_now:.2e} | "
              f"Train {tr_loss/len(train_loader):.4f}/{tr_acc:.1f}% | "
              f"Val Top-1 {live_top1:.1f}% Top-5 {live_top5:.1f}%")
        if live_tta_top1 is not None:
            print(f"  Live+TTA   Top-1 {live_tta_top1:.1f}% Top-5 {live_tta_top5:.1f}%")
        if ema_top1 is not None:
            print(f"  EMA        Top-1 {ema_top1:.1f}% Top-5 {ema_top5:.1f}%")
        if ema_tta_top1 is not None:
            print(f"  EMA+TTA    Top-1 {ema_tta_top1:.1f}% Top-5 {ema_tta_top5:.1f}%")

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
            torch.save((ema.module if ema is not None else model).state_dict(), model_path)
            print(f"  --> New best (selection metric): {best_acc:.1f}%")

if __name__ == "__main__":
    train()
