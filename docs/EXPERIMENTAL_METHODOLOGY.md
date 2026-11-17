# Experimental methodology

## Data

LoveDA uses eight labels and ignores label `0` in focal loss and mean IoU. LandCover.ai uses five classes—Background, Building, Woodland, Water, and Road—and uses `ignore_index=255`. Both pipelines resize images to 512x512 with bilinear interpolation and masks with nearest-neighbor interpolation.

## Models

- DeepLabV3+ with an ImageNet-pretrained ResNet-50 encoder.
- SegFormer-B2 initialized from `nvidia/mit-b2`.

SegFormer logits are bilinearly upsampled to mask dimensions before loss and metric computation.

## Optimization

Both models train for 50 epochs with AdamW, focal loss (`gamma=2.0`), per-GPU batch size 8, five linear warm-up epochs, cosine decay, and ImageNet normalization. ResNet-50 uses `1e-4 * world_size`; SegFormer uses `6e-5 * world_size`.

## Distributed execution

DDP uses NCCL, one process per GPU, `DistributedSampler`, and reductions for global loss and per-class metrics. Sync variants call `torch.nn.SyncBatchNorm.convert_sync_batchnorm`; no-sync variants retain local BatchNorm.

## Metrics and timing

The logs report per-class IoU, per-class accuracy, mean IoU, loss, and epoch time. The tables sum all 50 epoch times. Final-epoch time is reported separately because the final epoch also writes predictions and the checkpoint.

## Limitations

- One logged run is available per configuration.
- No CPU-only or eight-GPU log is present.
- SLURM node allocations differ between jobs.
- No repeated seeds or confidence intervals are available.
- The recorded logs are `.txt` files rather than `.out` files.
