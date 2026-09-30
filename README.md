# ERIS — Explainable, Risk-aware IDS for the Internet of Vehicles

This repo holds the **T5 CAN-bus intrusion detector**, the classification stage of ERIS. It extends the LLM-based integrated IDS of Aishwarya et al. (2025). The explainability and risk-assessment stages are in progress.


## What it does

Each CAN frame from **CICIoV2024** is turned into a short text sequence (`CAN_ID 1DC D0 02 D1 3D …`). T5 then generates one of six labels: `BENIGN`, `DoS`, `GAS`, `RPM`, `SPEED`, `STEERING_WHEEL`.

## Evaluation design

1. **Leak-free split.** Train, validation and test are split by *unique frame pattern*, so no frame appears in more than one split. The model is chosen on validation loss only; the test set is never used for selection.
2. **No formatting leaks.** The raw CSVs give away the class through formatting. The DoS file has no DLC column, and spoofing files write bytes unpadded (`0`, `2.0`, `000001DC`). All IDs and bytes are normalised to fixed-width uppercase hex, and constant fields are dropped.
3. **Pattern-level metrics** are reported alongside row-level metrics, because the dataset is highly repetitive. For example, GAS has only 2 distinct frames.
4. **Reload check.** The saved model is reloaded from disk and must give identical predictions.

## Results (`results/`)

t5-small architecture trained from scratch (45M params, 5 epochs; best epoch by validation loss = 3). Test set: 1,800 rows from held-out patterns.

| Metric | Value |
|---|---|
| Row accuracy | 91.7% |
| Macro F1 | 0.912 |
| Pattern-level accuracy | 97.0% (32/33) |

Every class except RPM scores 1.00 F1. All errors come from one RPM frame (`CAN ID 201`, predicted SPEED 149 times). Its payload is byte-for-byte identical to a SPEED frame (`06 1C 06 3F 06 2A 02 29`), which looks like a dataset artefact.

Pattern split per class: BENIGN 3,547 · DoS 21 · RPM 10 · SPEED 5 · STEERING_WHEEL 3 · GAS 2 (see `results/pattern_split.csv`). Classes with 1–3 test patterns give fragile per-class scores.

## Run it

Download the six `hexadecimal_*.csv` files of [CICIoV2024](https://www.unb.ca/cic/datasets/iov-dataset-2024.html) (not included here).

```bash
pip install torch transformers datasets scikit-learn pandas accelerate sentencepiece
python eris_t5.py --data /path/to/csvs            # pretrained t5-small (needs Hugging Face access)
python eris_t5.py --data /path/to/csvs --scratch  # same architecture, random init, hex word-level tokenizer
```

Or open `iovids.ipynb` in Google Colab (T4 GPU) and upload the CSVs to `/content`.

## Stack

PyTorch · Hugging Face Transformers / Datasets · scikit-learn · pandas
