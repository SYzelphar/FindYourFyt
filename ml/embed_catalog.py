"""Embed every catalog image (and product text, for CLIP models) with each encoder.

Outputs data/processed/embeddings/<encoder>_image.npy (and <encoder>_text.npy), rows in
catalog item_idx order, plus manifest.json.

Usage:
    python ml/embed_catalog.py [--encoders fashion_clip clip vgg16] [--batch-size 128]
"""
import argparse
import json
import time

import numpy as np
import polars as pl
import torch

import config as C
from encoders import item_text, load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(C.ENCODERS))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    catalog = pl.read_parquet(C.CATALOG).sort("item_idx")
    paths = [C.image_path(a) for a in catalog["article_id"]]
    texts = [item_text(r) for r in catalog.iter_rows(named=True)]
    C.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = C.EMBEDDINGS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(paths)} items on {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    for name in args.encoders:
        t0 = time.time()
        enc = load_encoder(name, device)
        img = enc.encode_images(paths, batch_size=args.batch_size, num_workers=args.workers)
        assert img.shape == (len(paths), enc.dim), img.shape
        np.save(C.EMBEDDINGS_DIR / f"{name}_image.npy", img)
        entry = {"model": C.ENCODERS[name], "dim": enc.dim, "items": len(paths), "image": True, "text": False}
        if enc.supports_text:
            np.save(C.EMBEDDINGS_DIR / f"{name}_text.npy", enc.encode_text(texts))
            entry["text"] = True
        entry["seconds"] = round(time.time() - t0, 1)
        manifest[name] = entry
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  {name}: done in {entry['seconds']}s")
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
