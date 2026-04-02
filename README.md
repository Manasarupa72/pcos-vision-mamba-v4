# PCOS Detection using Vision Mamba + SAM Segmentation (V4)

A scientifically rigorous deep learning pipeline for **Polycystic Ovary Syndrome (PCOS)** detection from ultrasound images using a custom **Vision Mamba** architecture with **SAM-based follicle segmentation**.

## 🔬 Key Features

- **Vision Mamba (V4):** Custom architecture combining ResNet-34 convolutional stem with Bidirectional Selective State Space Model (Mamba) blocks
- **Bidirectional Scanning:** Forward + reverse SSM scanning for richer spatial feature extraction
- **Stochastic Depth:** Linear drop-rate schedule (0→0.1) across 6 Mamba blocks for regularization
- **SAM Segmentation:** Segment Anything Model (SAM) for automatic follicle detection and Rotterdam criteria evaluation
- **5-Fold Stratified CV:** Rigorous cross-validation with a held-out 15% global test set
- **Publication-Ready Outputs:** Grad-CAM heatmaps, t-SNE visualizations, ablation studies, statistical significance tests, calibration analysis

## 📊 Results

| Metric | Test Set | 95% CI | CV Mean ± Std |
|---|---|---|---|
| Accuracy | See `results/results.json` | ✓ | ✓ |
| AUC-ROC | See `results/results.json` | ✓ | ✓ |
| F1-Score | See `results/results.json` | ✓ | ✓ |
| Sensitivity | See `results/results.json` | — | ✓ |
| Specificity | See `results/results.json` | — | ✓ |

> Full results including calibration metrics (ECE, Brier), ablation study, and statistical significance tests are in `results/`.

## 🏗️ Architecture

```
Input Image (256×256×3)
    │
    ▼
┌─────────────────────┐
│  ResNet-34 Conv Stem │  (pretrained feature extractor)
│  → 512ch × 8×8      │
└─────────┬───────────┘
          │ Flatten + Positional Embedding
          ▼
┌─────────────────────────────────────┐
│  6× Bidirectional Mamba Blocks      │
│  ├── Forward SSM  ──┐              │
│  ├── Backward SSM ──┤→ Average     │
│  └── Stochastic Depth (0→0.1)      │
└─────────┬───────────────────────────┘
          │ LayerNorm → Global Average Pooling
          ▼
┌─────────────────────┐
│  MLP Classifier     │
│  512 → 256 → 1      │
└─────────────────────┘
```

## 📁 Repository Structure

```
├── kaggle_notebook_v4.py        # Main pipeline (training + evaluation + SAM)
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── results/                     # Output plots, metrics, and analysis
│   ├── results.json             # All metrics (test, CV, calibration, ablation)
│   ├── fold_results.json        # Per-fold cross-validation results
│   ├── significance_tests.json  # McNemar, DeLong, bootstrap tests
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── threshold_curve.png
│   ├── per_class_accuracy.png
│   ├── reliability_diagram.png
│   ├── gradcam_gallery.png
│   ├── error_analysis.png
│   ├── tsne_features.png
│   ├── ablation_chart.png
│   ├── latex_table.tex
│   ├── test_results.csv
│   └── segmentation/           # SAM segmentation outputs
│       ├── follicle_counts.csv
│       ├── follicle_analysis.png
│       └── seg_*.png
└── .gitignore
```

## 🚀 How to Run

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (tested on Tesla T4, 16GB)
- PCOS ultrasound dataset with `infected/` and `noninfected/` subfolders

### On Kaggle (Recommended)

1. Upload your PCOS dataset as a Kaggle dataset
2. Create a new notebook, enable **GPU T4 x2**
3. Copy `kaggle_notebook_v4.py` into the notebook
4. Configure `SESSION_MODE`:

```python
# Session 1: Train folds 1-2
SESSION_MODE = 'train'
START_FOLD = 1
END_FOLD = 2

# Session 2: Train folds 3-4
SESSION_MODE = 'train'
START_FOLD = 3
END_FOLD = 4
PREV_CHECKPOINT_DIR = '/kaggle/input/your-session1-output/checkpoints'

# Session 3: Train fold 5
SESSION_MODE = 'train'
START_FOLD = 5
END_FOLD = 5
PREV_CHECKPOINT_DIR = '/kaggle/input/your-session2-output/checkpoints'

# Session 4: Full evaluation + SAM
SESSION_MODE = 'eval'
PREV_CHECKPOINT_DIR = '/kaggle/input/your-all-checkpoints/checkpoints'
```

5. Between sessions, download checkpoints from the **Output** tab and re-upload as a new dataset

### Full Run (if you have enough GPU time)

```python
SESSION_MODE = 'full'  # Trains all 5 folds + evaluation in one run
```

## 🧪 V4 Scientific Improvements

| Feature | Description |
|---|---|
| Data Integrity | MD5 hash-based deduplication before splitting |
| Cross-Validation | 5-fold stratified CV (replaces single 70/15/15 split) |
| Bidirectional SSM | Forward + reverse scanning averaged |
| Stochastic Depth | Linear 0→0.1 drop rate across Mamba blocks |
| Label Smoothing | ε = 0.05 for better calibration |
| CutMix + Mixup | Both applied randomly (25% each) during training |
| Ultrasound Augmentation | Speckle noise, shadow simulation, variable CLAHE |
| Calibration | ECE, Brier score, temperature scaling, reliability diagram |
| Statistical Tests | McNemar's, DeLong's AUC, paired bootstrap |
| Ablation Study | ±Bidirectional, ±Stochastic depth, block count variants |

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{pcos_vision_mamba_2026,
  title={PCOS Detection from Ultrasound Images using Vision Mamba with SAM-based Follicle Segmentation},
  year={2026}
}
```

## 📝 License

This project is for academic/research purposes.
