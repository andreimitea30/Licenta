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
FEATURES_DIR    = ROOT_DIR / "extracted_skeletons"
MODEL_SAVE_PATH = ROOT_DIR / "best_stgcn_model.pth"
CHECKPOINT_PATH = ROOT_DIR / "training_checkpoint.pth"

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
MIXUP_ALPHA   = 0.2
DROP_PATH_MAX = 0.3

FLIP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]

def normalize_skeleton(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    hip = (data[:, 23, :2] + data[:, 24, :2]) / 2
    data[:, :, :2] -= hip[:, np.newaxis, :]
    scale = np.linalg.norm(data[:, 11, :2] - data[:, 12, :2], axis=1).mean()
    if scale > 1e-4:
        data[:, :, :2] /= scale
    return data

def add_motion_features(data: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(data[:, :, :2])
    vel[1:] = data[1:, :, :2] - data[:-1, :, :2]
    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([data, vel, acc], axis=2)

def temporal_sample(data: np.ndarray, num_frames: int, augment: bool) -> np.ndarray:
    T = data.shape[0]
    if T >= num_frames:
        if augment:
            start = np.random.randint(0, T - num_frames + 1)
            return data[start: start + num_frames]
        idx = np.linspace(0, T - 1, num_frames, dtype=int)
        return data[idx]
    pad = np.repeat(data[-1:], num_frames - T, axis=0)
    return np.concatenate([data, pad], axis=0)

def augment_skeleton(data: np.ndarray) -> np.ndarray:
    data = data.copy()
    if np.random.random() < 0.5:
        data[:, :, 0] *= -1
        for l, r in FLIP_PAIRS:
            data[:, [l, r]] = data[:, [r, l]]
    if np.random.random() < 0.2:
        data = data[::-1].copy()
    scale = np.random.uniform(0.85, 1.15)
    data[:, :, :2] *= scale
    noise = np.random.randn(*data.shape).astype(np.float32)
    noise[:, :, 2] = 0
    data += noise * 0.03
    return data

class HAA500SkeletonDataset(Dataset):
    def __init__(self, split_dir, class_to_idx=None, max_frames=MAX_FRAMES, augment=False):
        self.max_frames = max_frames
        self.augment    = augment
        self.file_paths = sorted(glob.glob(
            os.path.join(split_dir, "**", "*.npy"), recursive=True))
        if class_to_idx is None:
            folders = sorted(glob.glob(os.path.join(split_dir, "*")))
            self.class_to_idx = {os.path.basename(f): i for i, f in enumerate(folders)}
        else:
            self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path       = self.file_paths[idx]
        class_name = os.path.basename(os.path.dirname(path))
        label      = self.class_to_idx[class_name]

        data = np.load(path).astype(np.float32)
        data = normalize_skeleton(data)
        data = temporal_sample(data, self.max_frames, self.augment)
        if self.augment:
            data = augment_skeleton(data)
        data = add_motion_features(data)

        x = np.transpose(data, (2, 0, 1))
        x = np.expand_dims(x, axis=-1)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)

class Graph:
    def __init__(self):
        self.num_node = 33
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
            (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
        ]
        self.A = self._build()

    def _build(self):
        A = np.zeros((self.num_node, self.num_node), dtype=np.float32)
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1
        A += np.eye(self.num_node)
        d = A.sum(1) ** -0.5
        A = np.diag(d) @ A @ np.diag(d)
        return torch.tensor(A, dtype=torch.float32).unsqueeze(0)

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer('A_fixed', A[0])
        self.A_learn = nn.Parameter(torch.zeros(A.shape[1], A.shape[2]))
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.einsum('nctv,vw->nctw', self.conv(x), self.A_fixed + self.A_learn)

class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, residual=True,
                 dropout=DROPOUT, drop_path_rate=0.0):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.gcn  = GraphConv(in_ch, out_ch, A)
        self.bn1  = nn.BatchNorm2d(out_ch)
        self.tcn  = nn.Conv2d(out_ch, out_ch, kernel_size=(9, 1),
                              stride=(stride, 1), padding=(4, 0))
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=False)

        if not residual:
            self.residual = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.training and self.drop_path_rate > 0:
            if torch.rand(1).item() < self.drop_path_rate:
                return self.relu(res) if not isinstance(res, int) else x
        out = self.drop(self.bn2(self.tcn(self.relu(self.bn1(self.gcn(x))))))
        return self.relu(out + res)

class STGCN(nn.Module):
    def __init__(self, num_classes=500, in_channels=3, dropout=DROPOUT,
                 drop_path_max=DROP_PATH_MAX):
        super().__init__()
        A   = Graph().A
        n   = 10
        dpr = [drop_path_max * i / (n - 1) for i in range(n)]

        self.data_bn = nn.BatchNorm1d(in_channels * 33)
        self.blocks  = nn.ModuleList([
            STGCNBlock(in_channels, 64,  A, residual=False, dropout=dropout, drop_path_rate=dpr[0]),
            STGCNBlock(64,  64,  A, dropout=dropout, drop_path_rate=dpr[1]),
            STGCNBlock(64,  64,  A, dropout=dropout, drop_path_rate=dpr[2]),
            STGCNBlock(64,  64,  A, dropout=dropout, drop_path_rate=dpr[3]),
            STGCNBlock(64,  128, A, stride=2, dropout=dropout, drop_path_rate=dpr[4]),
            STGCNBlock(128, 128, A, dropout=dropout, drop_path_rate=dpr[5]),
            STGCNBlock(128, 128, A, dropout=dropout, drop_path_rate=dpr[6]),
            STGCNBlock(128, 256, A, stride=2, dropout=dropout, drop_path_rate=dpr[7]),
            STGCNBlock(256, 256, A, dropout=dropout, drop_path_rate=dpr[8]),
            STGCNBlock(256, 256, A, dropout=dropout, drop_path_rate=dpr[9]),
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

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train():
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}  |  AMP: {use_amp}")
    if use_amp:
        torch.backends.cudnn.benchmark = True

    train_ds = HAA500SkeletonDataset(str(FEATURES_DIR / "train"), augment=True)
    val_ds   = HAA500SkeletonDataset(
        str(FEATURES_DIR / "val"), class_to_idx=train_ds.class_to_idx, augment=False)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=use_amp, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    model     = STGCN(num_classes=500, in_channels=IN_CHANNELS).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=BASE_LR * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch  = 0
    best_val_acc = 0.0

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        print(f"Resumed from epoch {start_epoch}  |  best val top-1: {best_val_acc:.1f}%")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        tr_loss = tr_correct = tr_total = 0

        for x, y in tqdm(train_loader, desc=f"E{epoch+1:03d}/{NUM_EPOCHS} train", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_mix, y_a, y_b, lam = mixup_data(x, y)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out  = model(x_mix)
                loss = mixup_loss(criterion, out, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            tr_loss    += loss.item()
            pred_y = y_a if lam >= 0.5 else y_b
            tr_correct += out.argmax(1).eq(pred_y).sum().item()
            tr_total   += y.size(0)

        scheduler.step()

        model.eval()
        v_loss = v_top1 = v_top5 = v_total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"E{epoch+1:03d}/{NUM_EPOCHS} val  ", leave=False):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out  = model(x)
                    loss = criterion(out, y)
                v_loss += loss.item()
                v_top1 += out.argmax(1).eq(y).sum().item()
                v_top5 += out.topk(5, dim=1)[1].eq(y.view(-1, 1)).any(dim=1).sum().item()
                v_total += y.size(0)

        tr_acc = 100.0 * tr_correct / tr_total
        top1   = 100.0 * v_top1 / v_total
        top5   = 100.0 * v_top5 / v_total
        lr_now = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | LR {lr_now:.2e} | "
            f"Train {tr_loss/len(train_loader):.4f} / {tr_acc:.1f}% | "
            f"Val {v_loss/len(val_loader):.4f} / Top-1 {top1:.1f}% / Top-5 {top5:.1f}%"
        )

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best_val_acc': best_val_acc,
        }, CHECKPOINT_PATH)

        if top1 > best_val_acc:
            best_val_acc = top1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> New best: {best_val_acc:.1f}%  (saved)")

if __name__ == "__main__":
    train()
