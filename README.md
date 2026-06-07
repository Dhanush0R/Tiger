# Assignment 3: TIGER — Generative Retrieval for Recommendation

Group 63 | 2526-S2 Recommender Systems, Leiden University

---

## Overview

This repository contains an implementation of TIGER (Transformer Index for GEnerative Recommenders) for sequential recommendation on the Amazon Toys & Games dataset. The pipeline consists of two training stages: an RQ-VAE that maps item content embeddings to discrete Semantic IDs, followed by an encoder-decoder Transformer that autoregressively predicts the next item's Semantic ID.

---

## Files

```
tiger_3.ipynb     — Main notebook (preprocessing → RQ-VAE → Transformer → evaluation → ablations)
report.pdf        — Experiment report
README.md         — This file
```

---

## Dataset

Download the Amazon Toys & Games files from [https://jmcauley.ucsd.edu/data/amazon/](https://jmcauley.ucsd.edu/data/amazon/) (2014 version):

- `ratings_Toys_and_Games.csv` — user–item interactions
- `meta_Toys_and_Games.json` — item metadata (title, description, categories, brand)

---

## Environment Setup (Google Colab T4)

1. Open [https://colab.research.google.com](https://colab.research.google.com) and upload `tiger_3.ipynb`.
2. Go to **Runtime → Change runtime type**, set **Hardware accelerator** to **GPU**, and select **T4**. Click **Save**.
3. Verify the GPU with `!nvidia-smi` — you should see an NVIDIA T4 (~15 GB).
4. Install dependencies by running the first notebook cell:

```python
!pip install sentence-transformers
```

---

## Google Drive Setup (one-time)

Create the following folder structure in your Google Drive before starting:

```
MyDrive/
└── tiger_assignment/
    ├── data/          ← upload the two dataset files here
    ├── embeddings/    ← Sentence-T5 embeddings will be cached here
    └── checkpoints/   ← RQ-VAE and Transformer checkpoints saved here
```

At the start of each Colab session, mount Drive and set the path constants at the top of the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = '/content/drive/MyDrive/tiger_assignment/data'
EMB_DIR  = '/content/drive/MyDrive/tiger_assignment/embeddings'
CKPT_DIR = '/content/drive/MyDrive/tiger_assignment/checkpoints'
```

---

## How to Reproduce

Run the notebook cells **top to bottom**. The notebook is divided into the following sections:

### 1. Data Preprocessing

Loads the interaction and metadata files, merges them on item ID, applies iterative 5-core filtering, builds chronological user sequences (capped at 20 items), and performs a leave-one-out train/val/test split.

### 2. Content Embeddings

Encodes each item's concatenated text fields (title, categories, brand, description) using `sentence-transformers/sentence-t5-base` (768-dim). Embeddings are saved to `EMB_DIR` and reloaded on subsequent sessions — **this step only needs to run once**.

### 3. RQ-VAE Training

Trains a Residual-Quantized VAE (L=3 codebook levels, K=256 codes each) to convert content embeddings into 3-token Semantic IDs. Checkpoints are saved to `CKPT_DIR/rqvae/` after each epoch. Once training converges, the model is frozen and Semantic IDs are extracted for all items.

To resume from a checkpoint, set `RESUME = True` at the top of the RQ-VAE training cell.

### 4. Transformer Training

Trains an encoder-decoder Transformer (4 encoder + 4 decoder layers, 6 heads, d_model=384, ff=1024) over flattened Semantic ID token sequences. Uses AdamW with linear warmup and early stopping on validation NDCG@10. Checkpoints are saved to `CKPT_DIR/transformer/`.

To resume training, set `RESUME = True` in the Transformer training cell.

### 5. Evaluation

Runs beam search (default beam size = 10) on the test set and reports **Recall@5**, **Recall@10**, **NDCG@5**, **NDCG@10** using full-item ranking. Invalid generations (Semantic IDs not matching any real item) are filtered out and their rate is reported.

### 6. Ablation Studies

Ablations are run over:
- RQ-VAE codebook size K (64, 128, 256)
- Number of RQ-VAE levels L (2, 3, 4)
- Transformer d_model (128, 256, 384)
- Number of Transformer layers (2, 4, 6)
- Number of attention heads (2, 4, 6)
- Beam size at inference (5, 10, 20, 30, 50)

Results are saved as CSV files in `CKPT_DIR/ablation/`.

---

## Reference

Rajput et al., "Recommender Systems with Generative Retrieval," NeurIPS 2023. [https://arxiv.org/abs/2305.05065](https://arxiv.org/abs/2305.05065)
