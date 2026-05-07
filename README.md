# TuringSketchLine Release Pipeline

This repository provides a pipeline for **manga sketch-to-line restoration** on **TuringSketchLine**.

The codebase is organized around a simple four-stage workflow:

1. build train/test splits and reference annotations  
2. train **SketchRet**  
3. generate predictions on the benchmark split  
4. evaluate the generated results  

The released pipeline is intended for benchmark reproduction and follow-up research on real draft-to-line restoration under sketch ambiguity and reference mismatch.

---

## Installation

We recommend using **Python 3.11** and installing PyTorch with CUDA first.  
Please install the project dependencies from `requirements.txt` first, and then install `xformers`.

```bash
# Create a new environment with Conda
conda create -n sketchret python=3.11
conda activate sketchret

# Install PyTorch first
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install project dependencies
pip install -r requirements.txt

# Install xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

---

## Repository Workflow

## Step 1. Build split and reference annotations

Starting from the released panel annotation file, first construct the training annotation file with reference pools and benchmark splits.

```bash
python tool/dataset_split.py \
  --release_jsonl /path/to/panel_index_en.jsonl \
  --crop_root /path/to/dataset_root \
  --output_root /path/to/dataset_root \
  --split_unit side_page \
  --test_ref_limit 1000 \
  --test_bucket_ratio 1:5:4
```

### Notes

- `panel_index_en.jsonl` is the released panel-level annotation file.
- `split_unit side_page` means splitting is performed at the **side-page** level to avoid leakage between training and testing.
- `test_bucket_ratio 1:5:4` controls the approximate composition of the test set over:
  - **no character**
  - **single character**
  - **multi-character**
- Since the split preserves side-page integrity, the final panel counts may be slightly different from the exact target. For example, with `--test_ref_limit 1000`, the selected test set may contain **1001** panels rather than exactly **1000**.

### Outputs

This step produces the following files:

- `panel_index_train_ref.jsonl`
- `splits/train.jsonl`
- `splits/test.jsonl`
- `refs/`
- `manifest_train_ref.json`

The generated `splits/test.jsonl` keeps **all selected test panels**, including panels without valid references, and should be used as the final benchmark test split.

---

## Step 2. Prepare the pretrained backbone

Before training, download or prepare a Stable Diffusion v1.5 checkpoint suitable for line-art generation, and set its path through `--pretrained_model_name_or_path`.

A typical local layout is:

```text
/path/to/pretrained_model/
├── scheduler/
├── tokenizer/
├── text_encoder/
├── vae/
└── unet/
```

---

## Step 3. Train the model

After the split file is prepared, train SketchRet using the generated training annotation file.

```bash
accelerate launch train.py \
  --pretrained_model_name_or_path /path/to/pretrained_model \
  --crop_root /path/to/dataset_root \
  --index_file /path/to/dataset_root/panel_index_train_ref.jsonl \
  --validation_index_file /path/to/dataset_root/splits/test.jsonl \
  --output_dir /path/to/output_dir
```

### Training notes

- `panel_index_train_ref.jsonl` is the main training annotation file.
- `splits/test.jsonl` is the predefined benchmark split used for validation preview and final testing.
- The training schedule uses three phases:
  - **Phase A**: sketch/control warmup
  - **Phase B**: retargeting warmup
  - **Phase C**: joint training

---

## Step 4. Generate predictions

After training, use the saved checkpoint to generate predictions on the predefined test split.

```bash
python eval/evaluate.py \
  --project_root /path/to/project_root \
  --checkpoint_path /path/to/output_dir/checkpoint-xxxx \
  --pretrained_model_name_or_path /path/to/pretrained_model \
  --crop_root /path/to/dataset_root \
  --index_file /path/to/dataset_root/splits/test.jsonl \
  --output_root /path/to/eval_output_root
```

### Outputs

This step is expected to produce prediction files together with a manifest for downstream evaluation, for example:

- `pred_panel/`
- `gt_panel/`
- `sketch_panel/`
- `manifest_release_eval.jsonl`

---

## Step 5. Evaluate the predictions

Finally, evaluate the generated results using the saved manifest file.

```bash
python eval/generate_samples.py \
  --manifest_jsonl /path/to/eval_output_root/manifest_release_eval.jsonl \
  --crop_root /path/to/dataset_root \
  --output_root /path/to/eval_output_root \
  --char_dino_model_name_or_path /path/to/dino_model
```

### Outputs

This step produces:

- `/path/to/eval_output_root/metrics_summary_release.json`
- `/path/to/eval_output_root/per_image_metrics_release.csv`

The summary file reports dataset-level metrics, while the CSV file stores per-sample evaluation results.

---

## Directory Layout

After the full pipeline is completed, the main files are organized as follows:

```text
/path/to/dataset_root/
├── panel_index_en.jsonl
├── panel_index_train_ref.jsonl
├── refs/
├── splits/
│   ├── train.jsonl
│   └── test.jsonl
└── manifest_train_ref.json

/path/to/output_dir/
├── logs/
└── checkpoint-xxxx

/path/to/eval_output_root/
├── pred_panel/
├── gt_panel/
├── sketch_panel/
├── manifest_release_eval.jsonl
├── metrics_summary_release.json
└── per_image_metrics_release.csv
```
