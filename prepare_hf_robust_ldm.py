import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser("Prepare HF dataset into D3 folder format")
    p.add_argument("--dataset_id", type=str, default="AniSundar18/Robust_LDM_Benchmark")
    p.add_argument("--output_root", type=str, default="data/robust_ldm_benchmark")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default=None, help="If absent, split train into train/val")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_key", type=str, default="image")
    p.add_argument("--label_key", type=str, default="label")
    p.add_argument("--real_labels", type=str, default="real,nature")
    p.add_argument("--fake_labels", type=str, default="fake,ai,ldm")
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_val", type=int, default=None)
    return p.parse_args()


def parse_label_list(s):
    return {x.strip().lower() for x in s.split(",") if x.strip()}


def image_to_pil(obj):
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            return Image.open(BytesIO(obj["bytes"])).convert("RGB")
        if obj.get("path") is not None:
            return Image.open(obj["path"]).convert("RGB")
    if isinstance(obj, str):
        return Image.open(obj).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(obj)}")


def make_label_mapper(ds_split, label_key, real_labels, fake_labels):
    names = None
    feature = ds_split.features.get(label_key, None)
    if hasattr(feature, "names"):
        names = feature.names

    def _map(example):
        v = example[label_key]
        if names is not None:
            name = names[v].lower()
        else:
            name = str(v).lower()

        if name in real_labels or "real" in name or "nature" in name:
            return "real"
        if name in fake_labels or "fake" in name or "ai" in name:
            return "fake"

        # Fallback for binary numeric labels
        if names is None and isinstance(v, int):
            if v == 0:
                return "real"
            if v == 1:
                return "fake"

        raise ValueError(f"Could not map label value={v}, name={name}")

    return _map


def export_split(ds_split, split_name, out_root, image_key, map_label, max_items=None):
    total = len(ds_split) if max_items is None else min(len(ds_split), max_items)
    saved = 0
    out_root = Path(out_root)
    for i, ex in enumerate(tqdm(ds_split, total=total, desc=f"[{split_name}]"), start=0):
        if max_items is not None and i >= max_items:
            break
        cls = map_label(ex)
        img = image_to_pil(ex[image_key])
        out_path = out_root / split_name / cls / f"{i:08d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        saved += 1
    return saved


def main():
    args = parse_args()
    real_labels = parse_label_list(args.real_labels)
    fake_labels = parse_label_list(args.fake_labels)

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ModuleNotFoundError(
            "HuggingFace datasets package is required. Install with `pip install datasets`."
        ) from e

    ds = load_dataset(args.dataset_id)
    if args.train_split not in ds:
        raise ValueError(f"Split `{args.train_split}` not found. Available: {list(ds.keys())}")

    train_ds = ds[args.train_split]
    if args.val_split is not None and args.val_split in ds:
        val_ds = ds[args.val_split]
    else:
        split = train_ds.train_test_split(
            test_size=args.val_ratio,
            seed=args.seed,
            stratify_by_column=args.label_key if args.label_key in train_ds.column_names else None,
        )
        train_ds = split["train"]
        val_ds = split["test"]

    mapper = make_label_mapper(train_ds, args.label_key, real_labels, fake_labels)

    out_root = Path(args.output_root)
    train_count = export_split(
        train_ds,
        "train",
        out_root,
        args.image_key,
        mapper,
        max_items=args.max_train,
    )
    val_count = export_split(
        val_ds,
        "val",
        out_root,
        args.image_key,
        mapper,
        max_items=args.max_val,
    )

    print(f"[DONE] saved train={train_count}, val={val_count} under {out_root}")
    print("Folder structure:")
    print(f"  {out_root}/train/real, {out_root}/train/fake")
    print(f"  {out_root}/val/real, {out_root}/val/fake")


if __name__ == "__main__":
    main()
