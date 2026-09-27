# SDQ

## Setup

1. Environment

```bash
conda create -n sdq python=3.10
conda activate sdq
python -m pip install --upgrade pip
python -m pip install -e .
freerec setup
```

2. Dataset


```bash
freerec make Amazon2014Beauty --root ../data --kcore4user 5 --kcore4item 5 --splitting LOU
```


3. Semantic embeddings

```bash
python encode_textual_features.py \
    --root ../data \
    --dataset Amazon2014Beauty_550_LOU \
    --model sentence-t5-xl \
    --model-dir ./models \
    --device cuda
```

This command writes `sentence-t5-xl_title_categories_brand.pkl` to the dataset directory. Set `--model-dir ""` to load `sentence-t5-xl` directly from Hugging Face instead of a local model directory. Select a PyTorch build compatible with the target CUDA runtime when preparing a GPU environment.

## Running SDQ

Run SDQ-VAE with the provided Beauty configuration:

```bash
python train_sdq_vae.py \
    --config configs/sdq/Amazon2014Beauty_550_LOU.yaml
```

Run the non-trainable SDQ-KMeans variant:

```bash
python train_sdq_kmeans.py \
    --dataset Amazon2014Beauty_550_LOU
```

## Training T5

T5 training requires the `sid_vocab.json` exported by an SDQ-VAE or SDQ-KMeans run. Specify its path through `--sid-vocab-file`:

```bash
python train_t5.py \
    --config configs/t5/Amazon2014Beauty_550_LOU.yaml \
    --sid-vocab-file logs/SDQ/Amazon2014Beauty_550_LOU/<run-id>/sid_vocab.json
```

Replace `<run-id>` with the identifier of the selected quantization run. For SDQ-KMeans, use the corresponding vocabulary under `logs/SDQ-KMeans/` instead.