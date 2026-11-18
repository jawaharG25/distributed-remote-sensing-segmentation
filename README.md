# Distributed Remote-Sensing Semantic Segmentation

Reproducible experiments studying GPU scaling and BatchNorm synchronization for semantic segmentation of high-resolution remote-sensing imagery.

## Project website

The thesis contents are presented as a scrolling React site in [`website/`](website/). It includes the abstract, datasets, architectures, DDP methodology, figures, scaling plot, complete result matrix, discussion, limitations, and direct links to the repository artifacts.

```powershell
cd website
npm install
npm run dev
```

The source is in [`website/src/main.jsx`](website/src/main.jsx) and [`website/src/styles.css`](website/src/styles.css).

## Research questions

- How does training time change from one to four GPUs?
- Does synchronized BatchNorm improve quality with small per-GPU batches?
- Do CNN and hierarchical Transformer models respond differently?
- Are the effects consistent across LoveDA and LandCover.ai?

## Experimental matrix

| Dataset | Models | GPUs | Ablation |
|---|---|---:|---|
| LoveDA | DeepLabV3+ / ResNet-50, SegFormer-B2 | 1, 2, 4 | SyncBatchNorm vs local BatchNorm |
| LandCover.ai | DeepLabV3+ / ResNet-50, SegFormer-B2 | 1, 2, 4 | SyncBatchNorm vs local BatchNorm |

All runs use 50 epochs, 512x512 inputs, per-GPU batch size 8, AdamW, focal loss (`gamma=2.0`), five warm-up epochs, and cosine learning-rate decay.

## Main recorded findings

- SegFormer-B2 has the higher final mIoU on both datasets: 0.5241 versus 0.5018 on LoveDA and 0.8609 versus 0.8358 on LandCover.ai at one GPU.
- Four-GPU scaling reduces summed logged time from 9,347.62 s to 2,452.44 s for LoveDA SegFormer-B2 and from 4,380.67 s to 1,123.08 s for LandCover.ai SegFormer-B2.
- SyncBatchNorm is not uniformly better than local BatchNorm; its effect is small and dataset/model dependent.
- LoveDA Barren and LandCover.ai Road are the hardest final classes by IoU.

See [results/tables/final_metrics.csv](results/tables/final_metrics.csv) and [results/tables/scaling_metrics.csv](results/tables/scaling_metrics.csv).

## Repository structure

```text
src/                  Training scripts, separated by dataset
slurm/                SLURM launch files, separated by dataset
results/logs/         Normalized copies of recorded logs
results/tables/       Machine-readable metrics
results/figures/      Curated qualitative and scaling collages
results/samples/      Representative prediction samples
docs/                 Methodology and reviewer guide
requirements.txt      Python dependencies
```

## Quick start

The scripts expect datasets and pretrained-model caches on the HPC filesystem. Update their dataset and output paths for a new environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Submit the relevant file under `slurm/` on the HPC system. Multi-GPU scripts use PyTorch DDP with NCCL.

## Artifact policy

Original datasets, generated bulk image directories, checkpoints, and environment caches are intentionally excluded because of size and possible redistribution restrictions. This repository retains source scripts, SLURM scripts, logs, result tables, collages, and representative samples.

## Reproducibility note

The current evidence contains one logged run per configuration. GPU node assignments and final-epoch output writing can affect wall-clock measurements; runtime results are observed job measurements rather than repeated benchmark estimates.
