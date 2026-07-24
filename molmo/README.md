# Molmo2 SPair-71k correspondence

Evaluates a local `Molmo2-8B` checkpoint on SPair-71k semantic point
correspondence with PCK@0.1. For each source keypoint, the source image is
marked with a red cross and Molmo2 is asked to emit the corresponding point in
the target image using its native multi-image `<points coords="...">` grammar.

This targets **Molmo2-8B**, not MolmoPoint-8B. MolmoPoint uses grounding tokens
and `extract_image_points()`, while regular Molmo2 emits textual coordinates.

## Environment

Ai2's model card specifies Python 3.11 and `transformers==4.57.1`:

```bash
conda create -n molmo2 python=3.11 -y
conda activate molmo2
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build separately if needed.

## Validate inputs without loading the model

```bash
python molmo_spair_correspondence.py \
  --dataset-root /data/shared-vilab/datasets/spair-71k/SPair-71k \
  --max-pairs 1 \
  --dry-run
```

Prepared source/target images and prompts are written under
`spair_correspondence_results/dry_run/`.

## Run

Small single-GPU test:

```bash
GPU_IDS=0 MAX_PAIRS=5 OVERWRITE=1 bash run_spair_correspondence.sh
```

Full split on four GPUs:

```bash
GPU_IDS=0,1,2,3 MAX_PAIRS=0 OVERWRITE=1 \
  SAVE_VISUALIZATIONS=1 bash run_spair_correspondence.sh
```

The run resumes existing JSONL output unless `OVERWRITE=1`. Outputs include:

- `test_predictions.jsonl`: per-keypoint results
- `test_summary.json`: PCK, parse rate, and per-category metrics
- `visualizations/test/`: source query, target GT, and prediction overlays
- `progress/shardXX.json`: live shard progress

Watch progress from another terminal:

```bash
python ../watch_spair_progress.py \
  --output-dir spair_correspondence_results
```

Useful environment variables include `MODEL_PATH`, `DATASET_ROOT`,
`OUTPUT_DIR`, `GPU_IDS`, `MAX_PAIRS`, `KEYPOINT_INDEX`,
`MIN_INPUT_PIXELS`, `MAX_NEW_TOKENS`, and `WATCH_PROGRESS`.

## References

- [Molmo2-8B model card](https://huggingface.co/allenai/Molmo2-8B)
- [Official Molmo2 repository](https://github.com/allenai/molmo2)
