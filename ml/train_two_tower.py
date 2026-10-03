"""Train the two-tower retrieval model on real H&M purchases.

Training examples: for every (customer, purchase day) in the train split, the customer's
previous purchases (last MAX_LEN distinct items, most recent first) predict each item
bought that day that is not already in the history. Full softmax over the whole catalog
(~30k items fit on the GPU, so no negative sampling is needed); history items are masked
out of the logits. Early stopping on validation Recall@12.

Outputs (data/processed/two_tower/): item_vectors.npy (catalog order), model.pt, vocab.json,
train_log.json.

Usage:
    python ml/train_two_tower.py [--epochs 10] [--batch-size 1024]
"""
import argparse
import json
import sys
import time

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F

import config as C

sys.path.insert(0, str(C.ROOT / "backend"))
from recsys.two_tower import TwoTower  # noqa: E402

CAT_COLUMNS = ["product_type_name", "colour_group_name", "graphical_appearance_name",
               "perceived_colour_master_name", "index_group_name", "garment_group_name", "section_name"]
MAX_LEN = 20


def item_features(catalog: pl.DataFrame):
    emb = C.EMBEDDINGS_DIR
    dense = np.concatenate([np.load(emb / "fashion_clip_image.npy"), np.load(emb / "fashion_clip_text.npy")], axis=1)
    vocab, cats = {}, []
    for col in CAT_COLUMNS:
        values = sorted(catalog[col].drop_nulls().unique().to_list())
        vocab[col] = {v: i + 1 for i, v in enumerate(values)}           # 0 = unknown
        cats.append([vocab[col].get(v, 0) for v in catalog[col].to_list()])
    return dense.astype(np.float32), np.array(cats, dtype=np.int64).T, vocab


def build_examples(tx: pl.DataFrame, max_len=MAX_LEN):
    """Return (hist [E, L] int32 padded -1, targets [E] int32) from a customer/day-sorted frame."""
    cust = tx["customer_idx"].to_numpy()
    day = tx["t_dat"].to_physical().to_numpy()
    item = tx["item_idx"].to_numpy()
    bounds = np.flatnonzero(np.diff(cust)) + 1
    hists, targets = [], []
    for c_items, c_days in zip(np.split(item, bounds), np.split(day, bounds)):
        if len(np.unique(c_days)) < 2:
            continue
        history: list[int] = []
        for d in np.unique(c_days):
            todays = c_items[c_days == d]
            if history:
                recent = list(dict.fromkeys(reversed(history)))[:max_len]
                hset = set(recent)
                row = recent + [-1] * (max_len - len(recent))
                for t in todays:
                    if t not in hset:
                        hists.append(row)
                        targets.append(t)
            history.extend(todays.tolist())
    return np.array(hists, dtype=np.int32), np.array(targets, dtype=np.int32)


def build_eval_set(history_tx: pl.DataFrame, target_tx: pl.DataFrame, max_customers, seed, max_len=MAX_LEN):
    """Per customer: history = last items before the period, targets = new items bought in it."""
    hist = (history_tx.sort("t_dat").group_by("customer_idx", maintain_order=True)
            .agg(pl.col("item_idx").reverse().unique(maintain_order=True).head(max_len).alias("hist")))
    tgt = target_tx.group_by("customer_idx").agg(pl.col("item_idx").unique().alias("targets"))
    df = hist.join(tgt, on="customer_idx", how="inner").sort("customer_idx")
    if df.height > max_customers:
        df = df.sample(max_customers, seed=seed)
    H = np.full((df.height, max_len), -1, dtype=np.int32)
    T = []
    for i, (h, t) in enumerate(zip(df["hist"].to_list(), df["targets"].to_list())):
        H[i, : len(h)] = h
        T.append(sorted(set(t) - set(h)))
    keep = [i for i, t in enumerate(T) if t]
    return H[keep], [T[i] for i in keep], df["customer_idx"].to_numpy()[keep]


def mask_history(logits: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
    """Set logits of items already in the history to -inf (padding = -1 is ignored)."""
    valid = (hist >= 0).to(torch.int32)
    counts = torch.zeros(logits.shape, dtype=torch.int32, device=logits.device)
    counts.scatter_add_(1, hist.clamp(min=0), valid)
    return logits.masked_fill(counts > 0, float("-inf"))


@torch.no_grad()
def recall_at_k(model, dense, cats, H, T, k=12, device="cuda", batch=4096):
    model.eval()
    E = model.item_tower(dense, cats)
    hits = total = 0
    for i in range(0, len(H), batch):
        h = torch.as_tensor(H[i:i + batch], device=device, dtype=torch.long)
        u = model.user_vectors(E, h)
        scores = mask_history((u @ E.T).float(), h)   # never recommend history items
        top = scores.topk(k, dim=1).indices.cpu().numpy()
        for row, targets in zip(top, T[i:i + batch]):
            hits += len(set(row.tolist()) & set(targets))
            total += len(targets)
    model.train()
    return hits / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = C.TWO_TOWER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = pl.read_parquet(C.CATALOG).sort("item_idx")
    dense_np, cats_np, vocab = item_features(catalog)
    dense = torch.as_tensor(dense_np, device=device)
    cats = torch.as_tensor(cats_np, device=device)

    tx = pl.read_parquet(C.TRANSACTIONS).sort(["customer_idx", "t_dat"])
    train_tx = tx.filter(pl.col("split") == "train")
    t0 = time.time()
    H, Y = build_examples(train_tx)
    print(f"{len(Y):,} training examples built in {time.time() - t0:.0f}s")
    Hv, Tv, _ = build_eval_set(train_tx, tx.filter(pl.col("split") == "val"), 20_000, C.SEED)
    print(f"{len(Hv):,} validation customers")

    model = TwoTower(dense.shape[1], [len(vocab[c]) for c in CAT_COLUMNS], dim=128, max_len=MAX_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(Y) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.05)
    H_t = torch.as_tensor(H, dtype=torch.long)
    Y_t = torch.as_tensor(Y, dtype=torch.long)

    log, best, bad = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(len(Y_t))
        losses = []
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            h, y = H_t[idx].to(device), Y_t[idx].to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                E = model.item_tower(dense, cats)
                u = model.user_vectors(E, h)
                logits = (u @ E.T).float() / model.log_tau.exp()
            logits = mask_history(logits, h)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                model.log_tau.clamp_(min=np.log(0.01), max=np.log(1.0))
            losses.append(loss.item())
        val_recall = recall_at_k(model, dense, cats, Hv, Tv, device=device)
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "val_recall@12": val_recall,
                 "tau": float(model.log_tau.detach().exp()), "seconds": round(time.time() - t0, 1)}
        log.append(entry)
        print(json.dumps(entry))
        if val_recall > best:
            best, bad = val_recall, 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        item_vecs = model.item_tower(dense, cats).float().cpu().numpy()
    np.save(out_dir / "item_vectors.npy", item_vecs.astype(np.float32))
    torch.save(model.user_tower.state_dict(), out_dir / "user_tower.pt")
    (out_dir / "vocab.json").write_text(json.dumps({"cat_columns": CAT_COLUMNS, "vocab": vocab, "max_len": MAX_LEN,
                                                    "dim": 128}, indent=1))
    (out_dir / "train_log.json").write_text(json.dumps({"best_val_recall@12": best, "examples": len(Y),
                                                        "val_customers": len(Hv), "epochs": log}, indent=2))
    print(f"best val Recall@12 = {best:.4f}; saved to {out_dir}")


if __name__ == "__main__":
    main()
