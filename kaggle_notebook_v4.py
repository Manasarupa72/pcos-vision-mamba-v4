"""
===========================================================================
PCOS Vision Mamba Classification + SAM Segmentation — Kaggle Notebook (V4)
===========================================================================
INSTRUCTIONS:
  1. Upload PCOS.zip as a Kaggle dataset
  2. Enable GPU: Settings → Accelerator → GPU T4 x2
  3. Run all cells

V4 CHANGES (Scientific Rigor Improvements):
  - Data integrity: hash-based deduplication check before splitting
  - 5-fold stratified cross-validation (replaces single 70/15/15 split)
  - Bidirectional Mamba scanning (forward + reverse averaging)
  - Stochastic depth in Mamba blocks (linear 0→0.1 drop rate)
  - Label smoothing (ε=0.05) + CutMix alongside Mixup
  - Ultrasound-specific augmentation (speckle noise, shadow, variable CLAHE)
  - Calibration: ECE, Brier score, temperature scaling, reliability diagram
  - Statistical significance: McNemar's, DeLong's AUC, paired bootstrap
  - Extended ablation: ±bidirectional, ±stochastic depth, ±SWA, block count
  - Vision Mamba only (standalone ResNet34 baseline removed)
  - All V3 outputs preserved (Grad-CAM, t-SNE, error analysis, SAM, etc.)
===========================================================================
"""

# ===================== CELL 1: Install Dependencies =======================
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'timm'])
import os
os.system(f'{sys.executable} -m pip install -q git+https://github.com/facebookresearch/segment-anything.git')
print("[OK] All packages installed (including SAM)!")

# ===================== CELL 2: All Imports ================================
import os
import random
import time
import json
import csv
import math
import hashlib
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    brier_score_loss
)
from sklearn.manifold import TSNE
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import timm

warnings.filterwarnings('ignore')

print("=" * 70)
print("PCOS Vision Mamba V4 — Scientifically Rigorous Pipeline")
print("=" * 70)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"GPU Memory: {mem / 1e9:.1f} GB")
    except Exception:
        pass
print("=" * 70)

# ===================== CONFIGURATION ======================================
def find_dataset_root():
    candidates = [
        '/kaggle/input/pcos/PCOS',
        '/kaggle/input/pcos',
        '/kaggle/input/pcos-dataset/PCOS',
        '/kaggle/input/pcos-zip/PCOS',
        '/kaggle/input',
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'infected')) and os.path.isdir(os.path.join(c, 'noninfected')):
            return c
    for root_dir in ['/kaggle/input']:
        if os.path.exists(root_dir):
            for dirpath, dirnames, filenames in os.walk(root_dir):
                if 'infected' in dirnames and 'noninfected' in dirnames:
                    return dirpath
    return None

DATASET_ROOT = find_dataset_root()

if DATASET_ROOT is None:
    print("[ERROR] Could not find dataset!")
    print("Looking for a folder containing 'infected/' and 'noninfected/' subfolders.")
    input_dir = '/kaggle/input'
    if os.path.exists(input_dir):
        for root, dirs, files in os.walk(input_dir):
            depth = root.replace(input_dir, '').count(os.sep)
            if depth > 2:
                dirs.clear()
                continue
            print(f"  {root}/ -> {dirs[:15]}")
    raise FileNotFoundError("Dataset not found! Make sure you added PCOS.zip as a Kaggle dataset.")

print(f"[OK] Dataset found at: {DATASET_ROOT}")

RESULTS_DIR = '/kaggle/working/results'
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 10
WARMUP_EPOCHS = 3
SWA_EPOCHS = 10
SWA_LR = 1e-5
SEED = 42
IMG_SIZE = 256
NUM_FOLDS = 5
TEST_RATIO = 0.15  # Global held-out test set
LABEL_SMOOTH_EPS = 0.05
STOCHASTIC_DEPTH_MAX = 0.1
N_MAMBA_BLOCKS = 6

# ===================== MULTI-SESSION CONFIG ================================
# SESSION_MODE controls what the notebook runs:
#   'train'  → Run folds START_FOLD through END_FOLD only, save checkpoints
#   'eval'   → Skip training, load all fold checkpoints, run evaluation + plots + SAM
#   'full'   → Original behavior: train all folds + evaluate (only if you have enough time)
SESSION_MODE = 'full'       # Change to 'train' or 'eval' per session
START_FOLD = 1              # First fold to train (1-indexed, inclusive)
END_FOLD = 5                # Last fold to train (1-indexed, inclusive)

# Path to checkpoints from previous session(s), uploaded as a Kaggle dataset
# Set to None if no prior checkpoints exist
PREV_CHECKPOINT_DIR = None  # e.g., '/kaggle/input/v4-session1/checkpoints'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Copy checkpoints from previous session(s) if available
# AUTO-DISCOVERY: scans ALL of /kaggle/input/ for checkpoint files
import shutil

def find_and_copy_checkpoints(checkpoint_dir, prev_dir=None):
    """Find checkpoint .pth files anywhere in /kaggle/input/ and copy to working dir."""
    expected_files = []
    for fold in range(1, NUM_FOLDS + 1):
        expected_files.append(f'best_model_fold{fold}.pth')
        expected_files.append(f'swa_model_fold{fold}.pth')

    copied_count = 0

    # Method 1: Try the explicit PREV_CHECKPOINT_DIR first
    if prev_dir and os.path.isdir(prev_dir):
        print(f"  [RESUME] Found checkpoint dir: {prev_dir}")
        for fname in os.listdir(prev_dir):
            if fname.endswith('.pth') or fname.endswith('.json'):
                src = os.path.join(prev_dir, fname)
                dst = os.path.join(checkpoint_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  [RESUME] Copied {fname}")
                    copied_count += 1

    # Method 2: Auto-scan /kaggle/input/ for any matching checkpoint files
    if copied_count == 0:
        print(f"  [RESUME] Auto-scanning /kaggle/input/ for checkpoint files...")
        input_dir = '/kaggle/input'
        if os.path.exists(input_dir):
            # First, list what's available for debugging
            print(f"  [DEBUG] /kaggle/input/ contents:")
            for item in sorted(os.listdir(input_dir)):
                item_path = os.path.join(input_dir, item)
                if os.path.isdir(item_path):
                    sub_items = os.listdir(item_path)
                    print(f"    /kaggle/input/{item}/ -> {sub_items[:15]}")

            # Walk all subdirectories looking for checkpoint files
            for dirpath, dirnames, filenames in os.walk(input_dir):
                for fname in filenames:
                    if fname in expected_files:
                        src = os.path.join(dirpath, fname)
                        dst = os.path.join(checkpoint_dir, fname)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            print(f"  [RESUME] Found & copied {fname} from {dirpath}")
                            copied_count += 1

    print(f"  [RESUME] Total checkpoint files copied: {copied_count}")
    if copied_count == 0:
        print(f"  [WARNING] No checkpoint files found! Make sure you added the dataset as Input.")
    return copied_count

find_and_copy_checkpoints(CHECKPOINT_DIR, PREV_CHECKPOINT_DIR)

# ===================== SEED =============================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== DATA INTEGRITY CHECK ===============================
print("\n[STEP 0/12] Data Integrity — Hash-Based Deduplication Check...")

def compute_image_hash(path):
    """MD5 hash of resized 64x64 image to detect duplicate content."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.resize(img, (64, 64))
        return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return None

def check_data_integrity(root_dir):
    """Check for duplicate images across classes."""
    classes = {'noninfected': 0, 'infected': 1}
    all_paths, all_labels, all_hashes = [], [], []
    hash_to_paths = {}
    for class_name, label in classes.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            paths = glob(os.path.join(class_dir, ext))
            for p in paths:
                h = compute_image_hash(p)
                if h is None:
                    continue
                all_paths.append(p)
                all_labels.append(label)
                all_hashes.append(h)
                if h not in hash_to_paths:
                    hash_to_paths[h] = []
                hash_to_paths[h].append((p, label))

    # Find duplicates
    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    cross_class_dupes = {h: paths for h, paths in duplicates.items()
                         if len(set(l for _, l in paths)) > 1}

    report = {
        'total_images': len(all_paths),
        'unique_hashes': len(hash_to_paths),
        'duplicate_groups': len(duplicates),
        'duplicate_images': sum(len(p) - 1 for p in duplicates.values()),
        'cross_class_duplicates': len(cross_class_dupes),
    }

    print(f"  Total images scanned:    {report['total_images']}")
    print(f"  Unique content hashes:   {report['unique_hashes']}")
    print(f"  Duplicate groups:        {report['duplicate_groups']}")
    print(f"  Extra duplicate images:  {report['duplicate_images']}")
    if cross_class_dupes:
        print(f"  [WARNING] Cross-class duplicates found: {len(cross_class_dupes)}")
        for h, paths in list(cross_class_dupes.items())[:3]:
            for p, l in paths:
                print(f"    {os.path.basename(p)} (label={l})")
    else:
        print("  [OK] No cross-class duplicate content detected.")

    # Deduplicate: keep first occurrence of each hash
    seen_hashes = set()
    deduped_paths, deduped_labels = [], []
    for p, l, h in zip(all_paths, all_labels, all_hashes):
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_paths.append(p)
            deduped_labels.append(l)

    print(f"  After dedup: {len(deduped_paths)} images "
          f"({sum(1 for l in deduped_labels if l==1)} infected, "
          f"{sum(1 for l in deduped_labels if l==0)} noninfected)")

    report['deduped_total'] = len(deduped_paths)
    return deduped_paths, deduped_labels, report

deduped_paths, deduped_labels, integrity_report = check_data_integrity(DATASET_ROOT)

with open(os.path.join(RESULTS_DIR, 'data_integrity_report.json'), 'w') as f:
    json.dump(integrity_report, f, indent=2)

# ===================== DATASET (ultrasound-specific augmentation) =========
class PCOSDataset(Dataset):
    def __init__(self, image_paths, labels, is_training=False):
        self.is_training = is_training
        self.image_paths = list(image_paths)
        self.labels = list(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if h != IMG_SIZE or w != IMG_SIZE:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        if self.is_training:
            image = self._augment(image, self.labels[idx])
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return image, label

    def _augment(self, image, label_val):
        """Ultrasound-specific augmentation. Non-PCOS (label=0) gets EXTRA augmentation."""
        # --- Common augmentation (both classes) ---
        # Variable CLAHE (randomized parameters)
        if random.random() < 0.5:
            try:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                clip_limit = random.uniform(1.0, 4.0)
                tile_size = random.choice([4, 8, 16])
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            except Exception:
                pass
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        if random.random() < 0.3:
            image = cv2.flip(image, 0)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))
        # Scale augmentation (zoom)
        if random.random() < 0.3:
            scale = random.uniform(0.85, 1.15)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (new_w, new_h))
            if scale > 1.0:
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                image = scaled[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                canvas = np.zeros_like(image)
                canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = scaled
                image = canvas
        if random.random() < 0.3:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        if random.random() < 0.4:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-10, 10)
            image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        # Speckle noise (ultrasound-specific)
        if random.random() < 0.3:
            sigma = random.uniform(0.05, 0.15)
            noise = np.random.randn(*image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) * (1.0 + sigma * noise), 0, 255).astype(np.uint8)
        # Gaussian noise
        elif random.random() < 0.2:
            noise = np.random.normal(0, 5, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Shadow simulation (ultrasound-specific)
        if random.random() < 0.2:
            h, w = image.shape[:2]
            shadow_y = random.randint(h // 4, 3 * h // 4)
            shadow_width = random.randint(h // 8, h // 4)
            shadow_mask = np.ones((h, w), dtype=np.float32)
            y1 = max(0, shadow_y - shadow_width // 2)
            y2 = min(h, shadow_y + shadow_width // 2)
            for y in range(y1, y2):
                dist = abs(y - shadow_y) / max(shadow_width / 2, 1)
                shadow_mask[y, :] = 0.3 + 0.7 * dist
            image = np.clip(image.astype(np.float32) * shadow_mask[:, :, np.newaxis],
                           0, 255).astype(np.uint8)

        # --- EXTRA augmentation for Non-PCOS only (label=0) ---
        if label_val == 0:
            if random.random() < 0.5:
                angle2 = random.uniform(-10, 10)
                h, w = image.shape[:2]
                M2 = cv2.getRotationMatrix2D((w / 2, h / 2), angle2, 1.0)
                image = cv2.warpAffine(image, M2, (w, h), borderValue=(0, 0, 0))
            if random.random() < 0.5:
                alpha2 = random.uniform(0.7, 1.3)
                beta2 = random.uniform(-20, 20)
                image = np.clip(alpha2 * image.astype(np.float32) + beta2, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                h, w = image.shape[:2]
                dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (7, 7), 3) * 4
                dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (7, 7), 3) * 4
                x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (x_grid + dx).astype(np.float32)
                map_y = (y_grid + dy).astype(np.float32)
                image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0))

        return image

# ===================== VISION MAMBA MODEL (V4: Bidirectional + Stochastic Depth) =
class DropPath(nn.Module):
    """Stochastic depth — drops the entire residual branch with probability p."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob


class SelectiveSSM(nn.Module):
    """Pure PyTorch implementation of Mamba's Selective State Space Model."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        ssm_params = self.x_proj(x_conv)
        B_param = ssm_params[:, :, :self.d_state]
        C_param = ssm_params[:, :, self.d_state:2*self.d_state]
        delta = F.softplus(ssm_params[:, :, -1:])
        delta = self.dt_proj(delta)
        delta = F.softplus(delta)
        A = -torch.exp(self.A_log)
        y = self._selective_scan(x_conv, delta, A, B_param, C_param)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, u, delta, A, B, C):
        batch_size, L, d_inner = u.shape
        N = A.shape[1]
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        h = torch.zeros(batch_size, d_inner, N, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = (h * C[:, i].unsqueeze(1)).sum(-1)
            ys.append(y)
        return torch.stack(ys, dim=1)


class BidirectionalMambaBlock(nn.Module):
    """Mamba block with bidirectional scanning and stochastic depth."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0.0,
                 bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        if bidirectional:
            self.mamba_bwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        normed = self.norm(x)
        fwd_out = self.mamba_fwd(normed)
        if self.bidirectional:
            bwd_out = self.mamba_bwd(normed.flip(1)).flip(1)
            ssm_out = (fwd_out + bwd_out) / 2.0
        else:
            ssm_out = fwd_out
        return x + self.drop_path(ssm_out)


class VimPCOS(nn.Module):
    """Vision Mamba (V4) — Bidirectional scanning + stochastic depth."""
    def __init__(self, num_classes=1, pretrained=True, n_mamba_blocks=6,
                 stochastic_depth_max=0.1, bidirectional=True):
        super().__init__()
        self.backbone_name = f"VisionMamba-V4-{'Bi' if bidirectional else 'Uni'}-{n_mamba_blocks}blk"
        resnet = timm.create_model('resnet34', pretrained=pretrained)
        self.conv_stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.act1, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            feat = self.conv_stem(dummy)
            _, C, H, W = feat.shape
            self.feat_h, self.feat_w = H, W
            self.seq_len = H * W
            self.feat_dim = C
        print(f"[MODEL] Conv stem output: {C}ch x {H}x{W} = {self.seq_len} seq len")

        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, self.feat_dim) * 0.02)
        # Stochastic depth: linearly increasing drop rates
        drop_rates = [stochastic_depth_max * i / max(n_mamba_blocks - 1, 1)
                      for i in range(n_mamba_blocks)]
        self.mamba_blocks = nn.ModuleList([
            BidirectionalMambaBlock(d_model=self.feat_dim, d_state=16, d_conv=4,
                                    expand=2, drop_path=dr, bidirectional=bidirectional)
            for dr in drop_rates
        ])
        bid_str = "Bidirectional" if bidirectional else "Unidirectional"
        print(f"[MODEL] {n_mamba_blocks} {bid_str} Mamba blocks, d_model={self.feat_dim}")
        print(f"[MODEL] Stochastic depth rates: {[f'{d:.3f}' for d in drop_rates]}")
        self.norm = nn.LayerNorm(self.feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        print(f"[MODEL] Backbone: {self.backbone_name}")

    def forward(self, x):
        x = self.conv_stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Extract features before classifier for t-SNE."""
        x = self.conv_stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# ===================== UTILITIES ========================================
def label_smooth(labels, eps=LABEL_SMOOTH_EPS):
    """Apply label smoothing: 0 -> eps, 1 -> 1-eps."""
    return labels * (1 - eps) + (1 - labels) * eps

def cutmix_data(x, y, alpha=1.0):
    """CutMix: paste a patch from one image onto another, mix labels by area."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_actual = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
    y_mixed = lam_actual * y + (1 - lam_actual) * y[idx]
    return x_mixed, y_mixed

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]

def evaluate_model(model, loader, device, criterion, threshold=0.5, use_tta=False):
    model.eval()
    all_labels, all_probs, all_logits = [], [], []
    total_loss, num_batches = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            if use_tta:
                probs_h = torch.sigmoid(model(torch.flip(images, [3])))
                probs_v = torch.sigmoid(model(torch.flip(images, [2])))
                probs_hv = torch.sigmoid(model(torch.flip(images, [2, 3])))
                probs = (probs + probs_h + probs_v + probs_hv) / 4.0
            all_logits.extend(logits.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))
            all_probs.extend(probs.cpu().numpy().flatten())
            total_loss += loss.item()
            num_batches += 1
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_logits = np.array(all_logits)
    all_preds = (all_probs >= threshold).astype(int)
    has_both = len(np.unique(all_labels)) > 1
    cm_eval = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn_e, fp_e, fn_e, tp_e = cm_eval.ravel() if cm_eval.size == 4 else (0, 0, 0, 0)
    sens = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0
    spec = tn_e / (tn_e + fp_e) if (tn_e + fp_e) > 0 else 0
    bal_acc = (sens + spec) / 2.0
    metrics = {
        'loss': total_loss / max(num_batches, 1),
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': bal_acc,
        'sensitivity': sens,
        'specificity': spec,
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_probs) if has_both else 0.0,
        'avg_precision': average_precision_score(all_labels, all_probs) if has_both else 0.0,
        'mcc': matthews_corrcoef(all_labels, all_preds),
    }
    return metrics, all_labels, all_probs, all_preds, all_logits

def find_optimal_threshold_balanced(model, loader, device, criterion, use_tta=False):
    _, val_labels, val_probs, _, _ = evaluate_model(model, loader, device, criterion,
                                                     threshold=0.5, use_tta=use_tta)
    best_thresh, best_bal_acc = 0.5, 0.0
    for thresh in np.arange(0.10, 0.91, 0.01):
        preds = (val_probs >= thresh).astype(int)
        cm_t = confusion_matrix(val_labels, preds, labels=[0, 1])
        if cm_t.size == 4:
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        else:
            continue
        sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
        bal_acc_t = (sens_t + spec_t) / 2.0
        if bal_acc_t > best_bal_acc:
            best_bal_acc = bal_acc_t
            best_thresh = thresh
    print(f"  [THRESHOLD] Optimal: {best_thresh:.2f} (BalAcc: {best_bal_acc:.4f})")
    return best_thresh, val_labels, val_probs

# ===================== CALIBRATION UTILITIES ==============================
def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(probs)

class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for calibration."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

def fit_temperature_scaling(val_logits, val_labels, device, n_iter=200, lr=0.01):
    """Fit temperature on validation set by minimizing NLL."""
    scaler = TemperatureScaler().to(device)
    logits_t = torch.tensor(val_logits, dtype=torch.float32).to(device)
    labels_t = torch.tensor(val_labels, dtype=torch.float32).to(device)
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=n_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled = scaler(logits_t)
        loss = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    print(f"  [CALIBRATION] Fitted temperature T = {scaler.temperature.item():.4f}")
    return scaler.temperature.item()

def plot_reliability_diagram(probs, labels, save_dir, title='Reliability Diagram', n_bins=15):
    """Plot reliability diagram showing calibration."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
            bin_counts.append(0)
        else:
            bin_accs.append(labels[mask].mean())
            bin_confs.append(probs[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i+1]) / 2 for i in range(n_bins)]
    width = 1.0 / n_bins * 0.8

    ax1.bar(bin_centers, bin_accs, width=width, alpha=0.7, color='#3498DB',
            edgecolor='black', label='Observed Frequency')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ece_val = compute_ece(probs, labels, n_bins)
    ax1.set_title(f'{title}\nECE = {ece_val:.4f}', fontsize=14)
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Observed Frequency', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.bar(bin_centers, bin_counts, width=width, color='#2ECC71', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'reliability_diagram.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: reliability_diagram.png")

# ===================== STATISTICAL SIGNIFICANCE ===========================
def mcnemar_test(labels, preds_a, preds_b):
    """McNemar's test for comparing two classifiers on the same data."""
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    b_val = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    c_val = np.sum(~correct_a & correct_b)  # A wrong, B correct
    n = b_val + c_val
    if n == 0:
        return 1.0, 0.0  # No discordant pairs
    # McNemar with continuity correction
    chi2 = (abs(b_val - c_val) - 1) ** 2 / max(n, 1)
    # p-value from chi2 distribution with 1 df
    from scipy import stats
    try:
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    except ImportError:
        # Fallback: approximate
        p_value = np.exp(-chi2 / 2)
    return p_value, chi2

def delong_auc_test(labels, probs_a, probs_b):
    """DeLong's test for comparing two AUCs from correlated samples."""
    n1 = np.sum(labels == 1)
    n0 = np.sum(labels == 0)
    pos_a = probs_a[labels == 1]
    neg_a = probs_a[labels == 0]
    pos_b = probs_b[labels == 1]
    neg_b = probs_b[labels == 0]
    # Structural components
    V_a10 = np.array([np.mean(pos_a > n) + 0.5 * np.mean(pos_a == n) for n in neg_a])
    V_a01 = np.array([np.mean(neg_a < p) + 0.5 * np.mean(neg_a == p) for p in pos_a])
    V_b10 = np.array([np.mean(pos_b > n) + 0.5 * np.mean(pos_b == n) for n in neg_b])
    V_b01 = np.array([np.mean(neg_b < p) + 0.5 * np.mean(neg_b == p) for p in pos_b])
    auc_a = np.mean(V_a10)
    auc_b = np.mean(V_b10)
    # Covariance matrix
    s10 = np.cov(np.stack([V_a10, V_b10]))
    s01 = np.cov(np.stack([V_a01, V_b01]))
    S = s10 / n0 + s01 / n1
    z = (auc_a - auc_b) / max(np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1]), 1e-10)
    from scipy import stats
    try:
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    except ImportError:
        p_value = np.exp(-z**2 / 2)
    return p_value, z, auc_a, auc_b

def paired_bootstrap_test(labels, probs_a, probs_b, n_boot=10000, seed=42):
    """Paired bootstrap test comparing two models' AUCs."""
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(labels), size=len(labels), replace=True)
        bl = labels[idx]
        if len(np.unique(bl)) < 2:
            continue
        auc_a = roc_auc_score(bl, probs_a[idx])
        auc_b = roc_auc_score(bl, probs_b[idx])
        diffs.append(auc_a - auc_b)
    diffs = np.array(diffs)
    p_value = np.mean(diffs <= 0) if np.mean(diffs) > 0 else np.mean(diffs >= 0)
    ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
    return p_value, np.mean(diffs), ci

# ===================== AMP HELPERS ======================================
def get_amp_autocast(device):
    if device.type != 'cuda':
        from contextlib import nullcontext
        return nullcontext()
    try:
        return torch.amp.autocast('cuda')
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast()

def get_grad_scaler(device):
    if device.type != 'cuda':
        return None
    try:
        return torch.amp.GradScaler('cuda')
    except (TypeError, AttributeError):
        try:
            return torch.amp.GradScaler()
        except (TypeError, AttributeError):
            return torch.cuda.amp.GradScaler()

def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

# ===================== PLOTTING =========================================
def save_plot(fig, filepath):
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(filepath)}")

def plot_training_curves(history, save_dir, fold_label=''):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    title_suffix = f' (Fold {fold_label})' if fold_label else ''
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title(f'Training & Validation Loss{title_suffix}', fontsize=13)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    axes[0, 1].set_title(f'Training & Validation Accuracy{title_suffix}', fontsize=13)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[0, 2].plot(epochs, history['val_bal_acc'], 'g-', label='Val Balanced Acc', linewidth=2)
    axes[0, 2].set_title(f'Validation Balanced Accuracy{title_suffix}', fontsize=13)
    axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('Balanced Accuracy')
    axes[0, 2].legend(); axes[0, 2].grid(True)
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train F1')
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val F1')
    axes[1, 0].set_title(f'Training & Validation F1-Score{title_suffix}', fontsize=13)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend(); axes[1, 0].grid(True)
    axes[1, 1].plot(epochs, history['val_auc'], 'r-', label='Val AUC-ROC', linewidth=2)
    axes[1, 1].set_title(f'Validation AUC-ROC{title_suffix}', fontsize=13)
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend(); axes[1, 1].grid(True)
    axes[1, 2].plot(epochs, history['lr'], 'm-', label='Learning Rate', linewidth=2)
    axes[1, 2].set_title(f'Learning Rate Schedule{title_suffix}', fontsize=13)
    axes[1, 2].set_xlabel('Epoch'); axes[1, 2].set_ylabel('LR')
    axes[1, 2].legend(); axes[1, 2].grid(True)
    plt.tight_layout()
    suffix = f'_fold{fold_label}' if fold_label else ''
    save_plot(fig, os.path.join(save_dir, f'training_curves{suffix}.png'))

def plot_confusion_matrix(cm, save_dir, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax)
    classes = ['Non-infected', 'Infected (PCOS)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12); ax.set_yticklabels(classes, fontsize=12)
    ax.set_ylabel('True Label', fontsize=12); ax.set_xlabel('Predicted Label', fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    save_plot(fig, os.path.join(save_dir, 'confusion_matrix.png'))

def plot_roc_curve(labels, probs, save_dir, ci_lower=None, ci_upper=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(8, 6))
    label_str = f'ROC Curve (AUC = {auc_val:.4f})'
    if ci_lower is not None and ci_upper is not None:
        label_str = f'ROC Curve (AUC = {auc_val:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])'
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=label_str)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Baseline')
    ax.set_title('ROC Curve', fontsize=14)
    ax.set_xlabel('False Positive Rate', fontsize=12); ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True)
    plt.tight_layout()
    save_plot(fig, os.path.join(save_dir, 'roc_curve.png'))

def plot_precision_recall_curve(labels, probs, save_dir):
    prec_arr, rec_arr, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_arr, prec_arr, 'g-', linewidth=2, label=f'PR Curve (AP = {ap:.4f})')
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
    ax.legend(fontsize=12); ax.grid(True)
    plt.tight_layout()
    save_plot(fig, os.path.join(save_dir, 'precision_recall_curve.png'))

def plot_threshold_curve(labels, probs, save_dir):
    thresholds = np.arange(0.05, 0.96, 0.01)
    sens_list, spec_list, bal_acc_list = [], [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        cm_t = confusion_matrix(labels, preds, labels=[0, 1])
        if cm_t.size == 4:
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
            s = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            sp = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
        else:
            s, sp = 0, 0
        sens_list.append(s); spec_list.append(sp)
        bal_acc_list.append((s + sp) / 2.0)
    best_idx = np.argmax(bal_acc_list)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, sens_list, 'r-', linewidth=2, label='Sensitivity (TPR)')
    ax.plot(thresholds, spec_list, 'b-', linewidth=2, label='Specificity (TNR)')
    ax.plot(thresholds, bal_acc_list, 'g-', linewidth=2.5, label='Balanced Accuracy')
    ax.axvline(x=thresholds[best_idx], color='black', linestyle='--', alpha=0.7,
               label=f'Optimal Threshold = {thresholds[best_idx]:.2f}')
    ax.set_xlabel('Threshold', fontsize=13); ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Sensitivity / Specificity / Balanced Accuracy vs Threshold', fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, os.path.join(save_dir, 'threshold_curve.png'))

def plot_per_class_accuracy(labels, preds, save_dir):
    cm_pc = confusion_matrix(labels, preds, labels=[0, 1])
    if cm_pc.size == 4:
        tn, fp, fn, tp = cm_pc.ravel()
        acc_non = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc_inf = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        acc_non, acc_inf = 0, 0
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['Non-infected\n(Specificity)', 'Infected / PCOS\n(Sensitivity)'],
                  [acc_non, acc_inf], color=['#4A90D9', '#E74C3C'], edgecolor='black', width=0.5)
    for bar, val in zip(bars, [acc_non, acc_inf]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1); ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Per-Class Accuracy', fontsize=14); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_plot(fig, os.path.join(save_dir, 'per_class_accuracy.png'))

# ===================== MAIN TRAINING PIPELINE (5-FOLD CV) =================
print("\n" + "=" * 70)
print("PCOS VISION MAMBA V4 — 5-FOLD CROSS-VALIDATION PIPELINE")
print("=" * 70)
print(f"Device: {device}")

# ---------- 1. GLOBAL TEST SET + K-FOLD SPLIT ----------
print("\n[STEP 1/12] Creating global test set + 5-fold CV splits...")
all_paths = np.array(deduped_paths)
all_labels = np.array(deduped_labels)

# Hold out 15% as global test set (never used during training)
sss_global = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=SEED)
trainval_idx, test_idx = next(sss_global.split(all_paths, all_labels))
test_paths = all_paths[test_idx].tolist()
test_labels = all_labels[test_idx].tolist()
trainval_paths = all_paths[trainval_idx]
trainval_labels = all_labels[trainval_idx]

print(f"  Global test set: {len(test_paths)} images "
      f"({sum(test_labels)} infected, {len(test_labels)-sum(test_labels)} noninfected)")
print(f"  Train+Val pool: {len(trainval_paths)} images")

# 5-fold stratified CV on the train+val pool
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
fold_splits = list(skf.split(trainval_paths, trainval_labels))

use_amp = device.type == 'cuda'

# ---------- 2. TRAIN EACH FOLD ----------
all_fold_metrics = []
best_fold_auc = 0.0
best_fold_idx = 0
best_fold_checkpoint = None

def get_lr(epoch, warmup_epochs=WARMUP_EPOCHS, max_epochs=EPOCHS, base_lr=LEARNING_RATE):
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
    fold_num = fold_idx + 1

    # --- Eval mode: skip ALL training ---
    if SESSION_MODE == 'eval':
        continue

    # --- Multi-session: skip folds not in this session's range ---
    if SESSION_MODE == 'train':
        if fold_num < START_FOLD or fold_num > END_FOLD:
            print(f"\n  [SKIP] Fold {fold_num} — outside session range [{START_FOLD}, {END_FOLD}]")
            continue

    # --- Skip folds that already have completed checkpoints ---
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_model_fold{fold_num}.pth')
    swa_ckpt_path = os.path.join(CHECKPOINT_DIR, f'swa_model_fold{fold_num}.pth')
    if os.path.exists(best_ckpt_path) and os.path.exists(swa_ckpt_path):
        print(f"\n  [SKIP] Fold {fold_num} — checkpoints already exist, loading metrics...")
        # Load this fold's metrics from checkpoint to track best fold
        ckpt_tmp = safe_torch_load(best_ckpt_path, device)
        fold_auc_saved = ckpt_tmp.get('val_auc', 0.0)
        if fold_auc_saved > best_fold_auc:
            best_fold_auc = fold_auc_saved
            best_fold_idx = fold_idx
            best_fold_checkpoint = f'best_model_fold{fold_num}.pth'
        del ckpt_tmp
        continue

    print(f"\n{'='*70}")
    print(f"  FOLD {fold_num}/{NUM_FOLDS}")
    print(f"{'='*70}")

    set_seed(SEED + fold_idx)

    fold_train_paths = trainval_paths[train_idx].tolist()
    fold_train_labels = trainval_labels[train_idx].tolist()
    fold_val_paths = trainval_paths[val_idx].tolist()
    fold_val_labels = trainval_labels[val_idx].tolist()

    print(f"  Train: {len(fold_train_paths)} | Val: {len(fold_val_paths)}")

    train_dataset = PCOSDataset(fold_train_paths, fold_train_labels, is_training=True)
    val_dataset = PCOSDataset(fold_val_paths, fold_val_labels, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Model
    model = VimPCOS(num_classes=1, pretrained=True, n_mamba_blocks=N_MAMBA_BLOCKS,
                    stochastic_depth_max=STOCHASTIC_DEPTH_MAX, bidirectional=True).to(device)
    if fold_num == 1:
        total_params, trainable_params = model.get_num_params()
        print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss with pos_weight + label smoothing
    num_pcos = sum(fold_train_labels)
    num_non = len(fold_train_labels) - num_pcos
    pos_weight_val = num_non / max(num_pcos, 1)
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_eval = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = get_grad_scaler(device) if use_amp else None

    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'val_auc': [], 'val_bal_acc': [], 'lr': []
    }
    best_val_auc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        current_lr = get_lr(epoch)
        set_lr(optimizer, current_lr)
        model.train()
        train_loss = 0
        train_labels_epoch, train_preds_epoch = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Label smoothing
            smooth_labels = label_smooth(labels, LABEL_SMOOTH_EPS)
            # Randomly choose Mixup or CutMix (50/50)
            r = random.random()
            if r < 0.25:
                images, smooth_labels = mixup_data(images, smooth_labels, alpha=0.4)
            elif r < 0.5:
                images, smooth_labels = cutmix_data(images, smooth_labels, alpha=1.0)
            optimizer.zero_grad()
            if use_amp and scaler is not None:
                with get_amp_autocast(device):
                    logits = model(images)
                    loss = criterion_train(logits, smooth_labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion_train(logits, smooth_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy().flatten()
            train_labels_epoch.extend(labels.cpu().numpy().flatten().astype(int))
            train_preds_epoch.extend(preds)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels_epoch, train_preds_epoch)
        train_f1 = f1_score(train_labels_epoch, train_preds_epoch, zero_division=0)
        val_metrics, _, _, _, _ = evaluate_model(model, val_loader, device, criterion_eval)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc_roc'])
        history['val_bal_acc'].append(val_metrics['balanced_accuracy'])
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1:
            print(f"  F{fold_num} Epoch {epoch:3d}/{EPOCHS} | "
                  f"Loss: {avg_train_loss:.4f}/{val_metrics['loss']:.4f} | "
                  f"Acc: {train_acc:.4f}/{val_metrics['accuracy']:.4f} | "
                  f"AUC: {val_metrics['auc_roc']:.4f} | {epoch_time:.1f}s")

        if val_metrics['auc_roc'] > best_val_auc:
            best_val_auc = val_metrics['auc_roc']
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc, 'fold': fold_num,
            }, os.path.join(CHECKPOINT_DIR, f'best_model_fold{fold_num}.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  [EARLY STOP] Fold {fold_num} stopped at epoch {epoch}")
                break

    fold_time = time.time() - start_time
    print(f"  Fold {fold_num} training: {fold_time:.1f}s | Best AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    # SWA for this fold
    print(f"  Fold {fold_num} SWA phase ({SWA_EPOCHS} epochs)...")
    ckpt = safe_torch_load(os.path.join(CHECKPOINT_DIR, f'best_model_fold{fold_num}.pth'), device)
    model.load_state_dict(ckpt['model_state_dict'])
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    for swa_epoch in range(1, SWA_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp and scaler is not None:
                with get_amp_autocast(device):
                    logits = model(images)
                    loss = criterion_train(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion_train(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        swa_model.update_parameters(model)
        swa_scheduler.step()
    update_bn(train_loader, swa_model, device=device)
    torch.save({'model_state_dict': swa_model.state_dict(), 'fold': fold_num},
               os.path.join(CHECKPOINT_DIR, f'swa_model_fold{fold_num}.pth'))

    # Evaluate fold on validation set
    fold_val_metrics, _, _, _, _ = evaluate_model(swa_model, val_loader, device, criterion_eval, use_tta=True)
    fold_val_metrics['fold'] = fold_num
    fold_val_metrics['best_epoch'] = best_epoch
    fold_val_metrics['train_time_s'] = round(fold_time, 1)
    all_fold_metrics.append(fold_val_metrics)

    # Save fold metrics incrementally (survives session timeout)
    with open(os.path.join(RESULTS_DIR, 'fold_results.json'), 'w') as f:
        json.dump({'folds': all_fold_metrics, 'partial': True}, f, indent=2)

    if fold_val_metrics['auc_roc'] > best_fold_auc:
        best_fold_auc = fold_val_metrics['auc_roc']
        best_fold_idx = fold_idx
        best_fold_checkpoint = f'best_model_fold{fold_num}.pth'

    print(f"  Fold {fold_num} final val: Acc={fold_val_metrics['accuracy']:.4f} "
          f"AUC={fold_val_metrics['auc_roc']:.4f} F1={fold_val_metrics['f1']:.4f}")

    # Save per-fold training curves
    plot_training_curves(history, RESULTS_DIR, fold_label=str(fold_num))

    # Clean up fold model
    del model, swa_model, train_dataset, val_dataset, train_loader, val_loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# ---------- 3. CROSS-VALIDATION SUMMARY ----------
if SESSION_MODE == 'train':
    # In train-only mode, just save what we have and exit
    print(f"\n{'='*70}")
    print(f"  SESSION COMPLETE — Folds {START_FOLD}–{END_FOLD} trained")
    print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"  Download these from 'Output' tab and upload as a dataset for next session")
    print(f"{'='*70}")
    # Save fold metrics collected so far
    if all_fold_metrics:
        with open(os.path.join(RESULTS_DIR, 'fold_results.json'), 'w') as f:
            json.dump({'folds': all_fold_metrics, 'partial': True}, f, indent=2)
    import sys; sys.exit(0)

# For 'eval' mode: load all fold metrics from checkpoints
if SESSION_MODE == 'eval':
    all_fold_metrics = []
    for fi in range(NUM_FOLDS):
        fn = fi + 1
        swa_path = os.path.join(CHECKPOINT_DIR, f'swa_model_fold{fn}.pth')
        best_path = os.path.join(CHECKPOINT_DIR, f'best_model_fold{fn}.pth')
        if not os.path.exists(swa_path) or not os.path.exists(best_path):
            print(f"  [WARNING] Missing checkpoints for Fold {fn} — skipping")
            continue
        # Quick evaluate this fold's SWA model on its val split
        fold_train_idx, fold_val_idx = fold_splits[fi]
        fold_val_paths = trainval_paths[fold_val_idx].tolist()
        fold_val_labels = trainval_labels[fold_val_idx].tolist()
        tmp_val_ds = PCOSDataset(fold_val_paths, fold_val_labels, is_training=False)
        tmp_val_loader = DataLoader(tmp_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        tmp_model = VimPCOS(num_classes=1, pretrained=False, n_mamba_blocks=N_MAMBA_BLOCKS,
                            stochastic_depth_max=STOCHASTIC_DEPTH_MAX, bidirectional=True).to(device)
        swa_tmp = AveragedModel(tmp_model)
        swa_ckpt = safe_torch_load(swa_path, device)
        swa_tmp.load_state_dict(swa_ckpt['model_state_dict'])
        criterion_eval = nn.BCEWithLogitsLoss()
        fold_metrics, _, _, _, _ = evaluate_model(swa_tmp, tmp_val_loader, device, criterion_eval, use_tta=True)
        fold_metrics['fold'] = fn
        best_ckpt = safe_torch_load(best_path, device)
        fold_metrics['best_epoch'] = best_ckpt.get('epoch', 0)
        all_fold_metrics.append(fold_metrics)
        if fold_metrics['auc_roc'] > best_fold_auc:
            best_fold_auc = fold_metrics['auc_roc']
            best_fold_idx = fi
            best_fold_checkpoint = f'best_model_fold{fn}.pth'
        del tmp_model, swa_tmp, tmp_val_ds, tmp_val_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"  Fold {fn}: AUC={fold_metrics['auc_roc']:.4f}")

print(f"\n{'='*70}")
print("  5-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*70}")

cv_metrics_keys = ['accuracy', 'balanced_accuracy', 'auc_roc', 'f1', 'sensitivity', 'specificity', 'mcc']
print(f"  {'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print(f"  {'-'*65}")
cv_summary = {}
for key in cv_metrics_keys:
    vals = [m[key] for m in all_fold_metrics]
    m, s = np.mean(vals), np.std(vals)
    cv_summary[key] = {'mean': float(m), 'std': float(s), 'min': float(np.min(vals)), 'max': float(np.max(vals))}
    print(f"  {key:<25} {m:>10.4f} {s:>10.4f} {np.min(vals):>10.4f} {np.max(vals):>10.4f}")

with open(os.path.join(RESULTS_DIR, 'fold_results.json'), 'w') as f:
    json.dump({'folds': all_fold_metrics, 'summary': cv_summary}, f, indent=2)

# ---------- 4. EVALUATE BEST FOLD ON GLOBAL TEST SET ----------
print(f"\n[STEP 4/12] Evaluating best fold (Fold {best_fold_idx+1}) on global test set...")
test_dataset = PCOSDataset(test_paths, test_labels, is_training=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

# Load best fold models for ensemble
best_fold_num = best_fold_idx + 1
model = VimPCOS(num_classes=1, pretrained=False, n_mamba_blocks=N_MAMBA_BLOCKS,
                stochastic_depth_max=STOCHASTIC_DEPTH_MAX, bidirectional=True).to(device)
total_params, trainable_params = model.get_num_params()
criterion_eval = nn.BCEWithLogitsLoss()

# Best-AUC checkpoint
ckpt_best = safe_torch_load(os.path.join(CHECKPOINT_DIR, f'best_model_fold{best_fold_num}.pth'), device)
model.load_state_dict(ckpt_best['model_state_dict'])
print(f"  Finding optimal threshold (best-AUC model, fold {best_fold_num})...")
# Rebuild val loader for best fold for threshold optimization
bf_train_idx, bf_val_idx = fold_splits[best_fold_idx]
bf_val_paths = trainval_paths[bf_val_idx].tolist()
bf_val_labels = trainval_labels[bf_val_idx].tolist()
bf_val_dataset = PCOSDataset(bf_val_paths, bf_val_labels, is_training=False)
bf_val_loader = DataLoader(bf_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

auc_thresh, _, _ = find_optimal_threshold_balanced(model, bf_val_loader, device, criterion_eval, use_tta=True)
_, _, probs_auc, _, logits_auc = evaluate_model(model, test_loader, device, criterion_eval, threshold=0.5, use_tta=True)

# SWA checkpoint
swa_model = AveragedModel(model)
ckpt_swa = safe_torch_load(os.path.join(CHECKPOINT_DIR, f'swa_model_fold{best_fold_num}.pth'), device)
swa_model.load_state_dict(ckpt_swa['model_state_dict'])
print(f"  Finding optimal threshold (SWA model, fold {best_fold_num})...")
swa_thresh, _, _ = find_optimal_threshold_balanced(swa_model, bf_val_loader, device, criterion_eval, use_tta=True)
_, _, probs_swa, _, logits_swa = evaluate_model(swa_model, test_loader, device, criterion_eval, threshold=0.5, use_tta=True)

# Ensemble
test_probs = (np.array(probs_swa) + np.array(probs_auc)) / 2.0
test_logits = (np.array(logits_swa) + np.array(logits_auc)) / 2.0

# Optimal ensemble threshold
_, val_labels_arr, val_probs_auc, _, _ = evaluate_model(model, bf_val_loader, device, criterion_eval, threshold=0.5, use_tta=True)
_, _, val_probs_swa, _, val_logits_swa = evaluate_model(swa_model, bf_val_loader, device, criterion_eval, threshold=0.5, use_tta=True)
val_probs_ens = (np.array(val_probs_auc) + np.array(val_probs_swa)) / 2.0
val_logits_ens = (np.array(val_probs_auc) + np.array(val_logits_swa)) / 2.0

best_thresh, best_bal_acc_val = 0.5, 0.0
for t in np.arange(0.10, 0.91, 0.01):
    preds_t = (val_probs_ens >= t).astype(int)
    cm_t = confusion_matrix(np.array(val_labels_arr), preds_t, labels=[0, 1])
    if cm_t.size == 4:
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
        bal_t = (sens_t + spec_t) / 2.0
        if bal_t > best_bal_acc_val:
            best_bal_acc_val = bal_t
            best_thresh = t
optimal_threshold = best_thresh
print(f"  [THRESHOLD] Ensemble optimal: {optimal_threshold:.2f} (val balanced acc: {best_bal_acc_val:.4f})")

test_preds = (test_probs >= optimal_threshold).astype(int)
test_labels_arr = np.array(test_labels)

# Final metrics
has_both = len(np.unique(test_labels_arr)) > 1
cm = confusion_matrix(test_labels_arr, test_preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
balanced_accuracy = (sensitivity + specificity) / 2.0

test_metrics = {
    'accuracy': accuracy_score(test_labels_arr, test_preds),
    'balanced_accuracy': balanced_accuracy,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'precision': precision_score(test_labels_arr, test_preds, zero_division=0),
    'recall': recall_score(test_labels_arr, test_preds, zero_division=0),
    'f1': f1_score(test_labels_arr, test_preds, zero_division=0),
    'auc_roc': roc_auc_score(test_labels_arr, test_probs) if has_both else 0.0,
    'avg_precision': average_precision_score(test_labels_arr, test_probs) if has_both else 0.0,
    'mcc': matthews_corrcoef(test_labels_arr, test_preds),
}

# ---------- 5. BOOTSTRAP CI ----------
print("\n[STEP 5/12] Bootstrap 95% Confidence Intervals (1000 iterations)...")
n_bootstrap = 1000
boot_aucs, boot_accs, boot_f1s = [], [], []
rng = np.random.RandomState(SEED)
for _ in range(n_bootstrap):
    idx = rng.choice(len(test_labels_arr), size=len(test_labels_arr), replace=True)
    bl = test_labels_arr[idx]
    bp = test_probs[idx]
    bpred = test_preds[idx]
    if len(np.unique(bl)) > 1:
        boot_aucs.append(roc_auc_score(bl, bp))
    boot_accs.append(accuracy_score(bl, bpred))
    boot_f1s.append(f1_score(bl, bpred, zero_division=0))

ci_auc = (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5))
ci_acc = (np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5))
ci_f1 = (np.percentile(boot_f1s, 2.5), np.percentile(boot_f1s, 97.5))
print(f"  AUC: {test_metrics['auc_roc']:.4f} [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
print(f"  Acc: {test_metrics['accuracy']:.4f} [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
print(f"  F1:  {test_metrics['f1']:.4f} [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")

# ---------- 6. CALIBRATION ----------
print("\n[STEP 6/12] Calibration Analysis (ECE, Brier, Temperature Scaling)...")
ece_before = compute_ece(test_probs, test_labels_arr)
brier_before = brier_score_loss(test_labels_arr, test_probs)
print(f"  ECE (before calibration):   {ece_before:.4f}")
print(f"  Brier Score (before):       {brier_before:.4f}")

# Temperature scaling on validation logits
val_labels_flat = np.array(bf_val_labels)
temperature = fit_temperature_scaling(val_logits_ens, val_labels_flat, device)
calibrated_probs = torch.sigmoid(torch.tensor(test_logits) / temperature).numpy()
ece_after = compute_ece(calibrated_probs, test_labels_arr)
brier_after = brier_score_loss(test_labels_arr, calibrated_probs)
print(f"  ECE (after calibration):    {ece_after:.4f}")
print(f"  Brier Score (after):        {brier_after:.4f}")

# Reliability diagram
plot_reliability_diagram(test_probs, test_labels_arr, RESULTS_DIR,
                          title='Reliability Diagram (Before Calibration)')

# ---------- 7. PRINT & SAVE RESULTS ----------
print("\n" + "=" * 70)
print("  FINAL TEST SET RESULTS (V4)")
print("=" * 70)

results_table = f"""
======================================================
          CLASSIFICATION PERFORMANCE (V4)
======================================================
  Accuracy                    {test_metrics['accuracy']:>10.4f}  [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]
  Balanced Accuracy           {test_metrics['balanced_accuracy']:>10.4f}
  Precision (PPV)             {test_metrics['precision']:>10.4f}
  Recall (Sensitivity/TPR)    {test_metrics['recall']:>10.4f}
  Specificity (TNR)           {specificity:>10.4f}
  F1-Score                    {test_metrics['f1']:>10.4f}  [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]
  AUC-ROC                     {test_metrics['auc_roc']:>10.4f}  [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]
  Average Precision (AP)      {test_metrics['avg_precision']:>10.4f}
  MCC                         {test_metrics['mcc']:>10.4f}
  Optimal Threshold           {optimal_threshold:>10.2f}
------------------------------------------------------
          CALIBRATION
------------------------------------------------------
  ECE (before / after):       {ece_before:.4f} / {ece_after:.4f}
  Brier (before / after):     {brier_before:.4f} / {brier_after:.4f}
  Temperature T:              {temperature:.4f}
------------------------------------------------------
          CROSS-VALIDATION ({NUM_FOLDS}-Fold)
------------------------------------------------------
  AUC Mean ± Std:             {cv_summary['auc_roc']['mean']:.4f} ± {cv_summary['auc_roc']['std']:.4f}
  Acc Mean ± Std:             {cv_summary['accuracy']['mean']:.4f} ± {cv_summary['accuracy']['std']:.4f}
  F1  Mean ± Std:             {cv_summary['f1']['mean']:.4f} ± {cv_summary['f1']['std']:.4f}
------------------------------------------------------
          CONFUSION MATRIX
------------------------------------------------------
                        Predicted
                    Non-inf   Infected
  True Non-inf     [{tn:>5}]     [{fp:>5}]
  True Infected    [{fn:>5}]     [{tp:>5}]
======================================================
"""
print(results_table)

print("\nDetailed Classification Report:")
print(classification_report(test_labels_arr, test_preds,
                            target_names=['Non-infected', 'Infected (PCOS)']))

# Save plots
print("[STEP 7/12] Saving all classification plots...")
plot_training_curves({'train_loss': all_fold_metrics[best_fold_idx].get('train_loss', []),
                       'val_loss': [], 'train_acc': [], 'val_acc': [],
                       'train_f1': [], 'val_f1': [], 'val_auc': [],
                       'val_bal_acc': [], 'lr': []}, RESULTS_DIR) if False else None
plot_confusion_matrix(cm, RESULTS_DIR, title=f'Confusion Matrix (V4 — {NUM_FOLDS}-Fold CV)')
plot_roc_curve(test_labels_arr, test_probs, RESULTS_DIR, ci_lower=ci_auc[0], ci_upper=ci_auc[1])
plot_precision_recall_curve(test_labels_arr, test_probs, RESULTS_DIR)
plot_threshold_curve(test_labels_arr, test_probs, RESULTS_DIR)
plot_per_class_accuracy(test_labels_arr, test_preds, RESULTS_DIR)

# Save JSON
results_json = {
    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
    'bootstrap_ci': {
        'auc_95ci': [float(ci_auc[0]), float(ci_auc[1])],
        'accuracy_95ci': [float(ci_acc[0]), float(ci_acc[1])],
        'f1_95ci': [float(ci_f1[0]), float(ci_f1[1])],
    },
    'calibration': {
        'ece_before': float(ece_before), 'ece_after': float(ece_after),
        'brier_before': float(brier_before), 'brier_after': float(brier_after),
        'temperature': float(temperature),
    },
    'cross_validation': cv_summary,
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'model_info': {
        'backbone': 'VisionMamba-V4-Bidirectional',
        'n_mamba_blocks': N_MAMBA_BLOCKS,
        'stochastic_depth_max': STOCHASTIC_DEPTH_MAX,
        'bidirectional': True,
        'label_smoothing': LABEL_SMOOTH_EPS,
        'total_params': total_params, 'trainable_params': trainable_params,
    },
    'dataset_info': {
        'total_images': len(deduped_paths),
        'test': len(test_paths),
        'num_folds': NUM_FOLDS,
        'data_integrity': integrity_report,
    },
    'training_config': {
        'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'max_epochs': EPOCHS,
        'swa_epochs': SWA_EPOCHS, 'seed': SEED,
    },
}

with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

# LaTeX table
latex_table = f"""
% ============ LaTeX Table (V4) ============
\\\\begin{{table}}[h]
\\\\centering
\\\\caption{{Classification Performance — Vision Mamba V4 ({NUM_FOLDS}-Fold CV, {len(deduped_paths)} images)}}
\\\\label{{tab:classification_results_v4}}
\\\\begin{{tabular}}{{lccc}}
\\\\toprule
\\\\textbf{{Metric}} & \\\\textbf{{Test}} & \\\\textbf{{95\\% CI}} & \\\\textbf{{CV Mean $\\\\pm$ Std}} \\\\\\\\
\\\\midrule
Accuracy & {test_metrics['accuracy']:.4f} & [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}] & {cv_summary['accuracy']['mean']:.4f} $\\\\pm$ {cv_summary['accuracy']['std']:.4f} \\\\\\\\
Balanced Accuracy & {test_metrics['balanced_accuracy']:.4f} & -- & {cv_summary['balanced_accuracy']['mean']:.4f} $\\\\pm$ {cv_summary['balanced_accuracy']['std']:.4f} \\\\\\\\
Sensitivity & {sensitivity:.4f} & -- & {cv_summary['sensitivity']['mean']:.4f} $\\\\pm$ {cv_summary['sensitivity']['std']:.4f} \\\\\\\\
Specificity & {specificity:.4f} & -- & {cv_summary['specificity']['mean']:.4f} $\\\\pm$ {cv_summary['specificity']['std']:.4f} \\\\\\\\
F1-Score & {test_metrics['f1']:.4f} & [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}] & {cv_summary['f1']['mean']:.4f} $\\\\pm$ {cv_summary['f1']['std']:.4f} \\\\\\\\
AUC-ROC & {test_metrics['auc_roc']:.4f} & [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}] & {cv_summary['auc_roc']['mean']:.4f} $\\\\pm$ {cv_summary['auc_roc']['std']:.4f} \\\\\\\\
MCC & {test_metrics['mcc']:.4f} & -- & {cv_summary['mcc']['mean']:.4f} $\\\\pm$ {cv_summary['mcc']['std']:.4f} \\\\\\\\
\\\\midrule
ECE (pre-cal.) & {ece_before:.4f} & -- & -- \\\\\\\\
ECE (post-cal.) & {ece_after:.4f} & -- & -- \\\\\\\\
Brier Score & {brier_before:.4f} & -- & -- \\\\\\\\
\\\\bottomrule
\\\\end{{tabular}}
\\\\end{{table}}
% ===================================================================
"""
print(latex_table)
with open(os.path.join(RESULTS_DIR, 'latex_table.tex'), 'w') as f:
    f.write(latex_table)

with open(os.path.join(RESULTS_DIR, 'test_results.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value', '95% CI Lower', '95% CI Upper', 'CV Mean', 'CV Std'])
    for key, val in test_metrics.items():
        ci = results_json['bootstrap_ci'].get(f'{key}_95ci', ['', ''])
        cv_m = cv_summary.get(key, {})
        writer.writerow([key, f"{val:.4f}",
                         ci[0] if ci[0] else '', ci[1] if ci[1] else '',
                         f"{cv_m.get('mean', ''):.4f}" if cv_m else '',
                         f"{cv_m.get('std', ''):.4f}" if cv_m else ''])

print(f"\nClassification results saved to: {RESULTS_DIR}")

# ===================== STEP 8: PUBLICATION FEATURES ======================

# ---------- 8a. GRAD-CAM HEATMAPS ----------
print("\n[STEP 8/12] Generating Grad-CAM Heatmaps...")

class GradCAM:
    """Simple Grad-CAM for the conv_stem."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=0):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        target = output[:, class_idx]
        target.backward(retain_graph=True)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

model.load_state_dict(ckpt_best['model_state_dict'])
model.eval()
target_layer = model.conv_stem[-1]
gradcam = GradCAM(model, target_layer)

tp_indices = [i for i in range(len(test_labels_arr)) if test_labels_arr[i] == 1 and test_preds[i] == 1]
tn_indices = [i for i in range(len(test_labels_arr)) if test_labels_arr[i] == 0 and test_preds[i] == 0]
fp_indices = [i for i in range(len(test_labels_arr)) if test_labels_arr[i] == 0 and test_preds[i] == 1]
fn_indices = [i for i in range(len(test_labels_arr)) if test_labels_arr[i] == 1 and test_preds[i] == 0]

n_samples = 3
categories = [
    ('True Positive (PCOS→PCOS)', tp_indices),
    ('True Negative (Non→Non)', tn_indices),
    ('False Positive (Non→PCOS)', fp_indices),
    ('False Negative (PCOS→Non)', fn_indices),
]

mean_norm = np.array([0.485, 0.456, 0.406])
std_norm = np.array([0.229, 0.224, 0.225])

fig, axes = plt.subplots(len(categories), n_samples * 2, figsize=(4 * n_samples * 2, 4 * len(categories)))
if len(categories) == 1:
    axes = axes.reshape(1, -1)

for row, (cat_name, indices) in enumerate(categories):
    selected = indices[:n_samples] if len(indices) >= n_samples else indices
    for col_idx, idx in enumerate(selected):
        img_path = test_paths[idx]
        img_orig = cv2.imread(img_path)
        if img_orig is not None:
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            img_orig = cv2.resize(img_orig, (IMG_SIZE, IMG_SIZE))
        else:
            img_orig = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img_tensor = torch.from_numpy(
            ((img_orig.astype(np.float32) / 255.0 - mean_norm) / std_norm).transpose(2, 0, 1)
        ).unsqueeze(0).float().to(device)
        cam = gradcam.generate(img_tensor)
        axes[row, col_idx * 2].imshow(img_orig)
        axes[row, col_idx * 2].set_title(f'{cat_name}\nProb: {test_probs[idx]:.3f}', fontsize=8)
        axes[row, col_idx * 2].axis('off')
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.5 * img_orig + 0.5 * heatmap)
        axes[row, col_idx * 2 + 1].imshow(overlay)
        axes[row, col_idx * 2 + 1].set_title('Grad-CAM', fontsize=8)
        axes[row, col_idx * 2 + 1].axis('off')
    for col_idx in range(len(selected), n_samples):
        axes[row, col_idx * 2].axis('off')
        axes[row, col_idx * 2 + 1].axis('off')

plt.suptitle('Grad-CAM Explainability — Vision Mamba V4', fontsize=14, fontweight='bold')
plt.tight_layout()
save_plot(fig, os.path.join(RESULTS_DIR, 'gradcam_gallery.png'))

# ---------- 8b. ERROR ANALYSIS ----------
print("[STEP 8b/12] Generating Error Analysis Gallery...")
misclassified = np.where(test_labels_arr != test_preds)[0]
error_probs = test_probs[misclassified]
sort_idx = np.argsort(-np.abs(error_probs - 0.5))
misclassified_sorted = misclassified[sort_idx]

n_errors_to_show = min(16, len(misclassified_sorted))
if n_errors_to_show > 0:
    n_cols = 4
    n_rows = (n_errors_to_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else (axes if n_errors_to_show > 1 else [axes])
    for i in range(n_errors_to_show):
        idx = misclassified_sorted[i]
        img = cv2.imread(test_paths[idx])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        else:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        true_name = 'PCOS' if test_labels_arr[idx] == 1 else 'Non-PCOS'
        pred_name = 'PCOS' if test_preds[idx] == 1 else 'Non-PCOS'
        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_name}\nPred: {pred_name} ({test_probs[idx]:.3f})',
                         fontsize=9, color='red')
        axes[i].axis('off')
    for i in range(n_errors_to_show, len(axes)):
        axes[i].axis('off')
    plt.suptitle(f'Error Analysis — {len(misclassified)} Misclassified (Top {n_errors_to_show})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, os.path.join(RESULTS_DIR, 'error_analysis.png'))
else:
    print("  No misclassified images!")
print(f"  Total errors: {len(misclassified)}/{len(test_labels_arr)} "
      f"({100*len(misclassified)/len(test_labels_arr):.1f}%)")

# ---------- 8c. STATISTICAL SIGNIFICANCE TESTS ----------
print("\n[STEP 8c/12] Statistical Significance Tests...")

# Train a quick unidirectional variant for comparison (10 epochs)
print("  Training unidirectional Mamba variant for significance comparison...")
ABLATION_EPOCHS = 10
uni_model = VimPCOS(num_classes=1, pretrained=True, n_mamba_blocks=N_MAMBA_BLOCKS,
                    stochastic_depth_max=STOCHASTIC_DEPTH_MAX, bidirectional=False).to(device)
uni_optimizer = optim.AdamW(uni_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
uni_scaler = get_grad_scaler(device) if use_amp else None
# Use best fold's training data
bf_train_paths = trainval_paths[bf_train_idx].tolist()
bf_train_labels = trainval_labels[bf_train_idx].tolist()
bf_train_dataset = PCOSDataset(bf_train_paths, bf_train_labels, is_training=True)
bf_train_loader = DataLoader(bf_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

num_pcos_bf = sum(bf_train_labels)
num_non_bf = len(bf_train_labels) - num_pcos_bf
pw_bf = torch.tensor([num_non_bf / max(num_pcos_bf, 1)], dtype=torch.float32).to(device)
criterion_abl = nn.BCEWithLogitsLoss(pos_weight=pw_bf)

for abl_epoch in range(1, ABLATION_EPOCHS + 1):
    abl_lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * abl_epoch / ABLATION_EPOCHS))
    set_lr(uni_optimizer, abl_lr)
    uni_model.train()
    for images, labels in bf_train_loader:
        images, labels = images.to(device), labels.to(device)
        uni_optimizer.zero_grad()
        if use_amp and uni_scaler is not None:
            with get_amp_autocast(device):
                logits = uni_model(images)
                loss = criterion_abl(logits, labels)
            uni_scaler.scale(loss).backward()
            uni_scaler.unscale_(uni_optimizer)
            torch.nn.utils.clip_grad_norm_(uni_model.parameters(), max_norm=1.0)
            uni_scaler.step(uni_optimizer)
            uni_scaler.update()
        else:
            logits = uni_model(images)
            loss = criterion_abl(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uni_model.parameters(), max_norm=1.0)
            uni_optimizer.step()
    if abl_epoch % 5 == 0:
        val_m, _, _, _, _ = evaluate_model(uni_model, bf_val_loader, device, criterion_eval)
        print(f"    Uni Epoch {abl_epoch}/{ABLATION_EPOCHS} | Val AUC: {val_m['auc_roc']:.4f}")

uni_metrics, _, uni_probs, uni_preds, _ = evaluate_model(uni_model, test_loader, device, criterion_eval, use_tta=True)
bi_preds = test_preds
bi_probs = test_probs

significance_results = {}

# McNemar's test
try:
    p_mcnemar, chi2_mcnemar = mcnemar_test(test_labels_arr, bi_preds, uni_preds)
    significance_results['mcnemar'] = {'p_value': float(p_mcnemar), 'chi2': float(chi2_mcnemar),
                                        'significant': bool(p_mcnemar < 0.05)}
    print(f"  McNemar's test: χ²={chi2_mcnemar:.4f}, p={p_mcnemar:.6f} "
          f"({'significant' if p_mcnemar < 0.05 else 'not significant'})")
except Exception as e:
    print(f"  McNemar's test skipped: {e}")

# DeLong's test
try:
    p_delong, z_delong, auc_bi, auc_uni = delong_auc_test(test_labels_arr, bi_probs, uni_probs)
    significance_results['delong'] = {'p_value': float(p_delong), 'z_statistic': float(z_delong),
                                       'auc_bidirectional': float(auc_bi), 'auc_unidirectional': float(auc_uni),
                                       'significant': bool(p_delong < 0.05)}
    print(f"  DeLong's test: Z={z_delong:.4f}, p={p_delong:.6f}, "
          f"AUC_bi={auc_bi:.4f} vs AUC_uni={auc_uni:.4f}")
except Exception as e:
    print(f"  DeLong's test skipped: {e}")

# Paired bootstrap
try:
    p_boot, delta_auc, ci_delta = paired_bootstrap_test(test_labels_arr, bi_probs, uni_probs)
    significance_results['paired_bootstrap'] = {'p_value': float(p_boot), 'delta_auc': float(delta_auc),
                                                 'delta_ci_95': [float(ci_delta[0]), float(ci_delta[1])]}
    print(f"  Paired bootstrap: ΔAUC={delta_auc:.4f} [{ci_delta[0]:.4f}, {ci_delta[1]:.4f}], p={p_boot:.6f}")
except Exception as e:
    print(f"  Paired bootstrap skipped: {e}")

with open(os.path.join(RESULTS_DIR, 'significance_tests.json'), 'w') as f:
    json.dump(significance_results, f, indent=2)

# ---------- 8d. t-SNE FEATURE VISUALIZATION ----------
print("\n[STEP 8d/12] Generating t-SNE Feature Visualization...")
model.load_state_dict(ckpt_best['model_state_dict'])
model.eval()

all_features, all_labels_tsne = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        feats = model.extract_features(images)
        all_features.append(feats.cpu().numpy())
        all_labels_tsne.extend(labels.numpy().flatten().astype(int))

all_features = np.concatenate(all_features, axis=0)
all_labels_tsne = np.array(all_labels_tsne)

tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, n_iter=1000)
features_2d = tsne.fit_transform(all_features)

fig, ax = plt.subplots(figsize=(10, 8))
for label, color, name in [(0, '#3498DB', 'Non-infected'), (1, '#E74C3C', 'Infected (PCOS)')]:
    mask = all_labels_tsne == label
    ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
              c=color, label=name, alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
ax.set_title('t-SNE Feature Visualization — Vision Mamba V4', fontsize=14)
ax.set_xlabel('t-SNE 1', fontsize=12); ax.set_ylabel('t-SNE 2', fontsize=12)
ax.legend(fontsize=12); ax.grid(True, alpha=0.2)
plt.tight_layout()
save_plot(fig, os.path.join(RESULTS_DIR, 'tsne_features.png'))

# ---------- 8e. EXTENDED ABLATION STUDY ----------
print("\n[STEP 9/12] Extended Ablation Study...")

ablation_configs = [
    ('Unidirectional\n(No BiMamba)', {'bidirectional': False, 'n_mamba_blocks': N_MAMBA_BLOCKS, 'stochastic_depth_max': STOCHASTIC_DEPTH_MAX}),
    ('No Stochastic\nDepth', {'bidirectional': True, 'n_mamba_blocks': N_MAMBA_BLOCKS, 'stochastic_depth_max': 0.0}),
    ('2 Mamba\nBlocks', {'bidirectional': True, 'n_mamba_blocks': 2, 'stochastic_depth_max': STOCHASTIC_DEPTH_MAX}),
    ('4 Mamba\nBlocks', {'bidirectional': True, 'n_mamba_blocks': 4, 'stochastic_depth_max': STOCHASTIC_DEPTH_MAX}),
]

ablation_results = {}

# Add main model results
ablation_results['Full V4\n(Bidirectional)'] = {
    'Accuracy': float(test_metrics['accuracy']),
    'AUC-ROC': float(test_metrics['auc_roc']),
    'Balanced Acc': float(test_metrics['balanced_accuracy']),
    'F1-Score': float(test_metrics['f1']),
}

# Add unidirectional results already computed
ablation_results['Unidirectional\n(No BiMamba)'] = {
    'Accuracy': float(uni_metrics['accuracy']),
    'AUC-ROC': float(uni_metrics['auc_roc']),
    'Balanced Acc': float(uni_metrics['balanced_accuracy']),
    'F1-Score': float(uni_metrics['f1']),
}

del uni_model
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Train remaining ablation variants
for abl_name, abl_kwargs in ablation_configs[1:]:  # Skip unidirectional (already done)
    print(f"  Training ablation: {abl_name.replace(chr(10), ' ')}...")
    abl_model = VimPCOS(num_classes=1, pretrained=True, **abl_kwargs).to(device)
    abl_opt = optim.AdamW(abl_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    abl_scl = get_grad_scaler(device) if use_amp else None
    for abl_ep in range(1, ABLATION_EPOCHS + 1):
        abl_lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * abl_ep / ABLATION_EPOCHS))
        set_lr(abl_opt, abl_lr)
        abl_model.train()
        for images, labels in bf_train_loader:
            images, labels = images.to(device), labels.to(device)
            abl_opt.zero_grad()
            if use_amp and abl_scl is not None:
                with get_amp_autocast(device):
                    logits = abl_model(images)
                    loss = criterion_abl(logits, labels)
                abl_scl.scale(loss).backward()
                abl_scl.unscale_(abl_opt)
                torch.nn.utils.clip_grad_norm_(abl_model.parameters(), max_norm=1.0)
                abl_scl.step(abl_opt)
                abl_scl.update()
            else:
                logits = abl_model(images)
                loss = criterion_abl(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(abl_model.parameters(), max_norm=1.0)
                abl_opt.step()
    abl_m, _, _, _, _ = evaluate_model(abl_model, test_loader, device, criterion_eval, use_tta=True)
    ablation_results[abl_name] = {
        'Accuracy': float(abl_m['accuracy']),
        'AUC-ROC': float(abl_m['auc_roc']),
        'Balanced Acc': float(abl_m['balanced_accuracy']),
        'F1-Score': float(abl_m['f1']),
    }
    print(f"    Acc={abl_m['accuracy']:.4f} AUC={abl_m['auc_roc']:.4f} F1={abl_m['f1']:.4f}")
    del abl_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# Plot ablation chart
abl_names = list(ablation_results.keys())
metrics_names = ['Accuracy', 'AUC-ROC', 'Balanced Acc', 'F1-Score']
colors_abl = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
x_abl = np.arange(len(abl_names))
width_abl = 0.18

fig, ax = plt.subplots(figsize=(16, 7))
for i, metric_name in enumerate(metrics_names):
    vals = [ablation_results[n][metric_name] for n in abl_names]
    bars = ax.bar(x_abl + i * width_abl, vals, width_abl, label=metric_name,
                  color=colors_abl[i], edgecolor='black')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xticks(x_abl + 1.5 * width_abl)
ax.set_xticklabels(abl_names, fontsize=9)
ax.set_ylim(0.80, 1.02)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Extended Ablation Study — V4 Component Analysis', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_plot(fig, os.path.join(RESULTS_DIR, 'ablation_chart.png'))

# Print ablation table
print("\n  Extended Ablation Results:")
print(f"  {'Configuration':<25} {'Accuracy':>10} {'AUC':>8} {'BalAcc':>8} {'F1':>8}")
print(f"  {'-'*60}")
for name, m in ablation_results.items():
    clean_name = name.replace('\n', ' ')
    print(f"  {clean_name:<25} {m['Accuracy']:>10.4f} {m['AUC-ROC']:>8.4f} {m['Balanced Acc']:>8.4f} {m['F1-Score']:>8.4f}")

results_json['ablation'] = {k.replace('\n', ' '): v for k, v in ablation_results.items()}
results_json['significance_tests'] = significance_results
with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)

del bf_train_dataset, bf_train_loader
if device.type == 'cuda':
    torch.cuda.empty_cache()

# ===================== STEP 10: SAM SEGMENTATION =========================
print("\n" + "=" * 70)
print("[STEP 10/12] SAM Follicle Segmentation & Counting...")
print("=" * 70)

SAM_CHECKPOINT = '/kaggle/working/sam_vit_b_01ec64.pth'
SAM_MODEL_TYPE = 'vit_b'
SEG_RESULTS_DIR = os.path.join(RESULTS_DIR, 'segmentation')
os.makedirs(SEG_RESULTS_DIR, exist_ok=True)

if not os.path.exists(SAM_CHECKPOINT):
    print("  Downloading SAM ViT-B checkpoint (~375 MB)...")
    import urllib.request
    sam_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
    urllib.request.urlretrieve(sam_url, SAM_CHECKPOINT)
    print(f"  Downloaded to {SAM_CHECKPOINT}")
else:
    print(f"  SAM checkpoint found: {SAM_CHECKPOINT}")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

print("  Loading SAM model...")
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device)
sam.eval()
print(f"  SAM {SAM_MODEL_TYPE} loaded on {device}")

mask_generator = SamAutomaticMaskGenerator(
    model=sam, points_per_side=32, pred_iou_thresh=0.86,
    stability_score_thresh=0.92, crop_n_layers=1,
    crop_n_points_downscale_factor=2, min_mask_region_area=50,
)

def filter_follicle_masks(masks, image_shape, min_area=80, max_area=8000,
                          min_circularity=0.25, min_solidity=0.4):
    h, w = image_shape[:2]
    total_area = h * w
    follicle_masks = []
    for mask_data in masks:
        seg = mask_data['segmentation']
        area = mask_data['area']
        if area < min_area or area > max_area or area > 0.15 * total_area:
            continue
        mask_uint8 = seg.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < 10:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * cnt_area / (perimeter * perimeter) if perimeter > 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = cnt_area / hull_area if hull_area > 0 else 0
        if circularity < min_circularity or solidity < min_solidity:
            continue
        mask_data['circularity'] = circularity
        mask_data['solidity'] = solidity
        follicle_masks.append(mask_data)
    return follicle_masks

def create_segmentation_overlay(image_rgb, follicle_masks, follicle_count,
                                 classification_label, classification_prob):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_rgb); axes[0].set_title('Original', fontsize=13); axes[0].axis('off')
    overlay = image_rgb.copy()
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(follicle_masks), 1)))
    for i, mask_data in enumerate(follicle_masks):
        seg = mask_data['segmentation']
        color = (np.array(colors[i % len(colors)][:3]) * 255).astype(np.uint8)
        mask_region = seg.astype(bool)
        overlay[mask_region] = (overlay[mask_region] * 0.5 + color * 0.5).astype(np.uint8)
        contour_mask = seg.astype(np.uint8) * 255
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, str(i+1), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    axes[1].imshow(overlay); axes[1].set_title(f'Segmentation ({follicle_count} follicles)', fontsize=13); axes[1].axis('off')
    combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for md in follicle_masks:
        combined_mask[md['segmentation']] = 255
    axes[2].imshow(combined_mask, cmap='hot')
    cls_text = "PCOS" if classification_label == 1 else "Non-PCOS"
    pcos_by_count = "Yes" if follicle_count >= 12 else "No"
    axes[2].set_title(f'Mask | {cls_text} ({classification_prob:.2f})\nFollicles: {follicle_count} | PCOS(>=12): {pcos_by_count}', fontsize=11)
    axes[2].axis('off')
    plt.tight_layout()
    return fig

print(f"\n  Processing {len(test_paths)} test images with SAM...")
seg_start_time = time.time()
follicle_results = []
max_vis_images = 30

for idx, (img_path, true_label) in enumerate(zip(test_paths, test_labels)):
    image = cv2.imread(img_path)
    if image is None:
        follicle_results.append({'image': os.path.basename(img_path), 'true_label': int(true_label),
            'classification_pred': int(test_preds[idx]), 'classification_prob': float(test_probs[idx]),
            'follicle_count': 0, 'pcos_by_follicle_count': False, 'error': 'Could not load'})
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (512, 512))
    try:
        with torch.no_grad():
            raw_masks = mask_generator.generate(image_resized)
    except Exception as e:
        follicle_results.append({'image': os.path.basename(img_path), 'true_label': int(true_label),
            'classification_pred': int(test_preds[idx]), 'classification_prob': float(test_probs[idx]),
            'follicle_count': 0, 'pcos_by_follicle_count': False, 'error': str(e)})
        continue
    follicle_masks = filter_follicle_masks(raw_masks, image_resized.shape)
    follicle_count = len(follicle_masks)
    result = {
        'image': os.path.basename(img_path), 'true_label': int(true_label),
        'true_label_name': 'infected' if true_label == 1 else 'noninfected',
        'classification_pred': int(test_preds[idx]), 'classification_prob': float(test_probs[idx]),
        'follicle_count': follicle_count, 'pcos_by_follicle_count': bool(follicle_count >= 12),
        'total_sam_masks': len(raw_masks),
    }
    follicle_results.append(result)
    if idx < max_vis_images:
        fig = create_segmentation_overlay(image_resized, follicle_masks, follicle_count,
            int(test_preds[idx]), float(test_probs[idx]))
        vis_path = os.path.join(SEG_RESULTS_DIR, f'seg_{idx:04d}_{os.path.basename(img_path)}')
        vis_path = os.path.splitext(vis_path)[0] + '.png'
        fig.savefig(vis_path, dpi=100, bbox_inches='tight'); plt.close(fig)
    if (idx + 1) % 20 == 0 or idx == 0:
        print(f"    [{idx+1}/{len(test_paths)}] {os.path.basename(img_path)} -> "
              f"{follicle_count} follicles | SAM masks: {len(raw_masks)}")

seg_time = time.time() - seg_start_time
print(f"\n  Segmentation completed in {seg_time:.1f}s ({seg_time/60:.1f} min)")

# Save follicle CSV
with open(os.path.join(SEG_RESULTS_DIR, 'follicle_counts.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'true_label', 'true_label_name', 'classification_pred',
                      'classification_prob', 'follicle_count', 'pcos_by_follicle_count', 'total_sam_masks'])
    for r in follicle_results:
        writer.writerow([r.get('image',''), r.get('true_label',''), r.get('true_label_name',''),
            r.get('classification_pred',''), f"{r.get('classification_prob',0):.4f}",
            r.get('follicle_count',0), r.get('pcos_by_follicle_count',False), r.get('total_sam_masks',0)])

# ===================== STEP 11: FOLLICLE ANALYSIS ========================
print("\n[STEP 11/12] Follicle Count Analysis & Rotterdam Criteria...")
counts = [r['follicle_count'] for r in follicle_results if 'error' not in r]
infected_counts = [r['follicle_count'] for r in follicle_results if 'error' not in r and r['true_label'] == 1]
noninfected_counts = [r['follicle_count'] for r in follicle_results if 'error' not in r and r['true_label'] == 0]

seg_summary = {
    'total_images': len(follicle_results),
    'avg_follicle_count': float(np.mean(counts)) if counts else 0,
    'avg_infected': float(np.mean(infected_counts)) if infected_counts else 0,
    'avg_noninfected': float(np.mean(noninfected_counts)) if noninfected_counts else 0,
    'pcos_by_count': sum(1 for r in follicle_results if r.get('pcos_by_follicle_count', False)),
    'seg_time_s': round(seg_time, 1),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if infected_counts and noninfected_counts:
    max_ct = max(max(infected_counts), max(noninfected_counts)) + 1
    bins = range(0, max_ct + 2)
    axes[0].hist(infected_counts, bins=bins, alpha=0.7, label='Infected (PCOS)', color='red', edgecolor='darkred')
    axes[0].hist(noninfected_counts, bins=bins, alpha=0.7, label='Non-infected', color='blue', edgecolor='darkblue')
    axes[0].axvline(x=12, color='green', linestyle='--', linewidth=2, label='Rotterdam Threshold (>=12)')
    axes[0].set_xlabel('Follicle Count', fontsize=12); axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].set_title('Follicle Count Distribution', fontsize=13); axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)

cls_probs = [r['classification_prob'] for r in follicle_results if 'error' not in r]
foll_counts = [r['follicle_count'] for r in follicle_results if 'error' not in r]
true_labels_seg = [r['true_label'] for r in follicle_results if 'error' not in r]
colors_scatter = ['red' if l == 1 else 'blue' for l in true_labels_seg]
axes[1].scatter(foll_counts, cls_probs, c=colors_scatter, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1].axhline(y=optimal_threshold, color='gray', linestyle='--', alpha=0.5, label=f'Cls Threshold ({optimal_threshold:.2f})')
axes[1].axvline(x=12, color='green', linestyle='--', alpha=0.5, label='Rotterdam (>=12)')
axes[1].set_xlabel('Follicle Count', fontsize=12); axes[1].set_ylabel('Cls Probability', fontsize=12)
axes[1].set_title('Classification Prob vs Follicle Count', fontsize=13); axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
save_plot(fig, os.path.join(SEG_RESULTS_DIR, 'follicle_analysis.png'))

# ===================== STEP 12: COMBINED RESULTS =========================
print("\n" + "=" * 70)
print("[STEP 12/12] Combined Results Summary")
print("=" * 70)

combined_report = f"""
{'='*70}
            COMBINED RESULTS SUMMARY (V4 — Publication-Ready)
{'='*70}

  CLASSIFICATION (Vision Mamba V4 — Bidirectional + Stochastic Depth)
  -----------------------------------------------
  Accuracy:           {test_metrics['accuracy']:.4f}  [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]
  Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}
  AUC-ROC:            {test_metrics['auc_roc']:.4f}  [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]
  F1-Score:           {test_metrics['f1']:.4f}  [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]
  Sensitivity:        {sensitivity:.4f}
  Specificity:        {specificity:.4f}

  CROSS-VALIDATION ({NUM_FOLDS}-Fold)
  -----------------------------------------------
  AUC:  {cv_summary['auc_roc']['mean']:.4f} ± {cv_summary['auc_roc']['std']:.4f}
  Acc:  {cv_summary['accuracy']['mean']:.4f} ± {cv_summary['accuracy']['std']:.4f}
  F1:   {cv_summary['f1']['mean']:.4f} ± {cv_summary['f1']['std']:.4f}

  CALIBRATION
  -----------------------------------------------
  ECE:   {ece_before:.4f} → {ece_after:.4f} (T={temperature:.4f})
  Brier: {brier_before:.4f} → {brier_after:.4f}

  SEGMENTATION (SAM {SAM_MODEL_TYPE})
  -----------------------------------------------
  Images Processed:   {seg_summary['total_images']}
  Avg Follicles:      {seg_summary['avg_follicle_count']:.1f}
  Avg (PCOS):         {seg_summary['avg_infected']:.1f}
  Avg (Non-PCOS):     {seg_summary['avg_noninfected']:.1f}
  PCOS by Count:      {seg_summary['pcos_by_count']}/{seg_summary['total_images']}
{'='*70}
"""
print(combined_report)

results_json['segmentation'] = seg_summary
results_json['follicle_results'] = follicle_results
with open(os.path.join(RESULTS_DIR, 'combined_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)
with open(os.path.join(SEG_RESULTS_DIR, 'segmentation_summary.json'), 'w') as f:
    json.dump(seg_summary, f, indent=2)

print(f"\n{'='*70}")
print("ALL DONE! V4 Scientifically Rigorous Pipeline Complete.")
print(f"{'='*70}")
print(f"\n  Classification results: {RESULTS_DIR}/")
print(f"    - training_curves_foldN.png     (per-fold training)")
print(f"    - confusion_matrix.png")
print(f"    - roc_curve.png                 (with 95% CI)")
print(f"    - precision_recall_curve.png")
print(f"    - threshold_curve.png")
print(f"    - per_class_accuracy.png")
print(f"    - reliability_diagram.png       (NEW — calibration)")
print(f"    - gradcam_gallery.png           (Grad-CAM heatmaps)")
print(f"    - error_analysis.png            (misclassified gallery)")
print(f"    - tsne_features.png             (feature space)")
print(f"    - ablation_chart.png            (extended ablation)")
print(f"    - results.json                  (all metrics + calibration + ablation)")
print(f"    - fold_results.json             (NEW — 5-fold CV results)")
print(f"    - significance_tests.json       (NEW — McNemar/DeLong/bootstrap)")
print(f"    - test_results.csv              (with CV mean ± std)")
print(f"    - latex_table.tex               (with CV + CI + calibration)")
print(f"\n  Segmentation results: {SEG_RESULTS_DIR}/")
print(f"    - follicle_counts.csv")
print(f"    - follicle_analysis.png")
print(f"    - segmentation_summary.json")
print(f"    - seg_XXXX_*.png")
print(f"\n  Combined: {RESULTS_DIR}/combined_results.json")
print(f"\nDownload results from the 'Output' tab on the right.")
