"""
ERIS - T5 CAN-bus intrusion detector

Pipeline
  1. CSVs read as strings; hex normalised (IDs -> 3-digit, bytes -> 2-digit uppercase).
     Constant fields (Interface, DLC) are dropped - the DoS file has no DLC column,
     so keeping it would leak the label.
  2. Train / validation / test split by unique frame pattern (no frame in two splits);
     model selected on validation loss only.
  3. Metrics at row level and unique-pattern level.
  4. Saved model reloaded from disk and re-checked.

Usage
  python eris_t5.py --data "<folder with hexadecimal_*.csv>" [--scratch]
  --scratch : no Hugging Face access -> same t5-small architecture, random init,
              word-level hex tokenizer. Default: pretrained t5-small (needs HF).
"""
import argparse, json, os, random
import numpy as np, pandas as pd, torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", default=None)
p.add_argument("--scratch", action="store_true")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--train_per_class", type=int, default=1400)
p.add_argument("--eval_per_class", type=int, default=300)
args = p.parse_args()
OUT = args.out or os.path.join(args.data, "results")
os.makedirs(OUT, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

FILES = {
    "BENIGN": "hexadecimal_benign.csv",
    "DoS": "hexadecimal_DoS.csv",
    "GAS": "hexadecimal_spoofing-GAS.csv",
    "RPM": "hexadecimal_spoofing-RPM.csv",
    "SPEED": "hexadecimal_spoofing-SPEED.csv",
    "STEERING_WHEEL": "hexadecimal_spoofing-STEERING_WHEEL.csv",
}
DATA_COLS = [f"DATA_{i}" for i in range(8)]

# ---------------------------------------------------------------- 1. load + normalise
def norm_byte(v):  # "2.0" / "2" / "02" -> "02"
    return "%02X" % int(str(v).replace(".0", ""), 16)

frames = []
for cls, fn in FILES.items():
    df = pd.read_csv(os.path.join(args.data, fn), dtype=str, keep_default_na=False,
                     usecols=["ID"] + DATA_COLS)
    df["ID"] = df["ID"].map(lambda v: "%03X" % int(v, 16))
    for c in DATA_COLS:
        df[c] = df[c].map(norm_byte)
    df["cls"] = cls
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
data["utr"] = "CAN_ID " + data["ID"]
for i, c in enumerate(DATA_COLS):
    data["utr"] = data["utr"] + f" D{i} " + data[c]
del frames

owners = data.groupby("utr")["cls"].nunique()
assert (owners == 1).all(), "a UTR belongs to >1 class"

# ---------------------------------------------------------------- 2. pattern-level split
split_of = {}
pattern_table = []
for cls, g in data.groupby("cls"):
    pats = sorted(g["utr"].unique())
    random.Random(SEED).shuffle(pats)
    n = len(pats)
    if n >= 3:
        n_te = max(1, round(0.15 * n)); n_va = max(1, round(0.15 * n))
    elif n == 2:
        n_te, n_va = 1, 0
    else:
        n_te, n_va = 0, 0
    te, va, tr = pats[:n_te], pats[n_te:n_te + n_va], pats[n_te + n_va:]
    for s, lst in (("train", tr), ("val", va), ("test", te)):
        for u in lst:
            split_of[u] = s
    pattern_table.append({"class": cls, "patterns": n, "train": len(tr), "val": len(va), "test": len(te)})
data["split"] = data["utr"].map(split_of)
print(pd.DataFrame(pattern_table).to_string(index=False))

def sample(split, k):
    parts = []
    for cls, g in data[data["split"] == split].groupby("cls"):
        parts.append(g.sample(n=min(k, len(g)), random_state=SEED))
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

train_df = sample("train", args.train_per_class)
val_df = sample("val", args.eval_per_class)
test_df = sample("test", args.eval_per_class)
del data
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    ov = set(locals()[f"{a}_df"]["utr"]) & set(locals()[f"{b}_df"]["utr"])
    assert not ov, f"UTR overlap {a}/{b}"
print("\nrows  train/val/test:", len(train_df), len(val_df), len(test_df))
print("train:", train_df["cls"].value_counts().to_dict())
print("val  :", val_df["cls"].value_counts().to_dict())
print("test :", test_df["cls"].value_counts().to_dict())

# ---------------------------------------------------------------- 3. tokenizer + model
from transformers import (T5Config, T5ForConditionalGeneration, PreTrainedTokenizerFast,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from datasets import Dataset

CLASSES = list(FILES)
if args.scratch:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    vocab_list = ["<pad>", "</s>", "<unk>", "CAN_ID"] + [f"D{i}" for i in range(8)] + \
                 ["%02X" % b for b in range(256)] + ["%03X" % i for i in range(0x800)] + CLASSES
    vocab = {t: i for i, t in enumerate(dict.fromkeys(vocab_list))}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>",
                                        eos_token="</s>", unk_token="<unk>")
    cfg = T5Config(vocab_size=len(vocab), d_model=512, d_kv=64, d_ff=2048, num_layers=6,
                   num_decoder_layers=6, num_heads=8, pad_token_id=0, eos_token_id=1,
                   decoder_start_token_id=0)  # = t5-small architecture
    model = T5ForConditionalGeneration(cfg)
    lr = 3e-4
else:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    lr = 3e-4
print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

def to_ds(df):
    return Dataset.from_pandas(df[["utr", "cls"]].rename(columns={"utr": "text", "cls": "label"}),
                               preserve_index=False)

def tok(ex):
    enc = tokenizer(ex["text"], max_length=64, truncation=True)
    enc["labels"] = tokenizer(text_target=ex["label"], max_length=8, truncation=True)["input_ids"]
    return enc

train_tok = to_ds(train_df).map(tok, batched=True, remove_columns=["text", "label"])
val_tok = to_ds(val_df).map(tok, batched=True, remove_columns=["text", "label"])

device = "cuda" if torch.cuda.is_available() else "cpu"
targs = Seq2SeqTrainingArguments(
    output_dir=os.path.join(OUT, "ckpt"), num_train_epochs=args.epochs, learning_rate=lr,
    per_device_train_batch_size=16, per_device_eval_batch_size=64,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    fp16=(device == "cuda"), logging_steps=50, report_to="none", seed=SEED,
    predict_with_generate=False)
trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_tok, eval_dataset=val_tok,
                         data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                         processing_class=tokenizer)  # eval_dataset = VALIDATION, not test
trainer.train()
print("best checkpoint (by val loss):", trainer.state.best_model_checkpoint)

FINAL = os.path.join(OUT, "model")
trainer.save_model(FINAL); tokenizer.save_pretrained(FINAL)

# ---------------------------------------------------------------- 4. evaluate on TEST
def predict(m, texts, bs=64):
    m.eval().to(device); out = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], return_tensors="pt", padding=True,
                        truncation=True, max_length=64).to(device)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=6)
        out += [s.strip() for s in tokenizer.batch_decode(g, skip_special_tokens=True)]
    return out

texts = test_df["utr"].tolist()
y_true = test_df["cls"].tolist()
y_pred = predict(trainer.model, texts)

# reload from disk, predictions must match
reloaded = T5ForConditionalGeneration.from_pretrained(FINAL)
same = predict(reloaded, texts) == y_pred
print("reloaded checkpoint gives identical predictions:", same)

labels = sorted(set(y_true) | set(y_pred))
acc = accuracy_score(y_true, y_pred)
pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)

res = test_df[["utr", "cls"]].rename(columns={"cls": "true_class"}).copy()
res["predicted_class"] = y_pred
pat = res.drop_duplicates("utr")
pat_acc = (pat["true_class"] == pat["predicted_class"]).mean()
pat_by_cls = pat.assign(ok=pat["true_class"] == pat["predicted_class"]) \
                .groupby("true_class")["ok"].agg(["sum", "count"])

report = classification_report(y_true, y_pred, zero_division=0, digits=4)
print("\n=== TEST (held-out patterns) ===")
print(f"row accuracy {acc:.4f} | macro P/R/F1 {pm:.4f}/{rm:.4f}/{fm:.4f} | weighted F1 {fw:.4f}")
print(f"pattern-level accuracy {pat_acc:.4f} ({int(pat_by_cls['sum'].sum())}/{len(pat)} unique UTRs)")
print(report); print(cm); print(pat_by_cls)

res.to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)
cm.to_csv(os.path.join(OUT, "confusion_matrix.csv"))
pd.DataFrame(pattern_table).to_csv(os.path.join(OUT, "pattern_split.csv"), index=False)
json.dump({"mode": "scratch" if args.scratch else "pretrained t5-small",
           "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
           "row_accuracy": acc, "macro_precision": pm, "macro_recall": rm, "macro_f1": fm,
           "weighted_precision": pw, "weighted_recall": rw, "weighted_f1": fw,
           "pattern_accuracy": pat_acc, "pattern_correct": int(pat_by_cls["sum"].sum()),
           "pattern_total": int(len(pat)), "reload_identical": same,
           "best_checkpoint": trainer.state.best_model_checkpoint,
           "log_history": trainer.state.log_history},
          open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=str)
open(os.path.join(OUT, "classification_report.txt"), "w").write(report + "\n\n" + cm.to_string())
print("saved to", OUT)
