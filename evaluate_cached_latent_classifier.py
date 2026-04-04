import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.latent_pair_classification_dataset import LatentPairClassificationDataset
from models.latent_classifier import build_latent_pair_classifier
from train_cached_latent_classifier import calculate_split_acc, find_best_threshold


@torch.no_grad()
def collect_predictions(model, loader, device: str):
    model.eval()
    y_true, y_pred, rel_paths = [], [], []
    for z_clean, z_aux, labels, rel in loader:
        z_clean = z_clean.to(device, non_blocking=True)
        z_aux = z_aux.to(device, non_blocking=True)
        logits = model(z_clean, z_aux)
        probs = logits.sigmoid().flatten().cpu().numpy()
        y_pred.extend(probs.tolist())
        y_true.extend(labels.numpy().astype(np.int64).tolist())
        rel_paths.extend(rel)
    return np.asarray(y_true), np.asarray(y_pred), rel_paths


def average_precision(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.float64)
    order = np.argsort(-y_pred, kind='mergesort')
    y_true = y_true[order]
    positives = (y_true == 1)
    n_pos = int(positives.sum())
    if n_pos == 0:
        return 0.0
    tp = np.cumsum(positives)
    precision = tp / (np.arange(len(y_true)) + 1.0)
    return float((precision * positives).sum() / n_pos)


def metrics_from_predictions(y_true, y_pred, fixed_threshold=None):
    ap = average_precision(y_true, y_pred)
    real_acc05, fake_acc05, acc05, balanced_acc05 = calculate_split_acc(y_true, y_pred, 0.5)
    best_threshold = find_best_threshold(y_true, y_pred)
    real_best, fake_best, best_acc, best_balanced = calculate_split_acc(y_true, y_pred, best_threshold)
    metrics = {
        'ap': ap,
        'acc@0.5': acc05,
        'real_acc@0.5': real_acc05,
        'fake_acc@0.5': fake_acc05,
        'balanced_acc@0.5': balanced_acc05,
        'best_threshold_on_split': best_threshold,
        'best_acc_on_split': best_acc,
        'best_real_acc_on_split': real_best,
        'best_fake_acc_on_split': fake_best,
        'best_balanced_acc_on_split': best_balanced,
        'num_samples': int(y_true.size),
        'num_real': int((y_true == 0).sum()),
        'num_fake': int((y_true == 1).sum()),
    }
    if fixed_threshold is not None:
        real_fixed, fake_fixed, acc_fixed, balanced_fixed = calculate_split_acc(y_true, y_pred, fixed_threshold)
        metrics.update({
            'fixed_threshold': float(fixed_threshold),
            'acc@fixed_threshold': acc_fixed,
            'real_acc@fixed_threshold': real_fixed,
            'fake_acc@fixed_threshold': fake_fixed,
            'balanced_acc@fixed_threshold': balanced_fixed,
        })
    return metrics


def build_loader(dataset, batch_size, num_workers, prefetch_factor, device):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith('cuda'),
        drop_last=False,
    )
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def get_generators(clean_root: Path, split: str):
    split_root = clean_root / split
    return sorted(path.name for path in split_root.iterdir() if path.is_dir() and not path.name.startswith('.'))


def evaluate_split(model, clean_root, aux_root, split, include_path_contains, batch_size, num_workers, prefetch_factor, device):
    dataset = LatentPairClassificationDataset(
        clean_latent_root=str(clean_root),
        aux_latent_root=str(aux_root),
        split=split,
        include_path_contains=include_path_contains,
    )
    loader = build_loader(dataset, batch_size, num_workers, prefetch_factor, device)
    y_true, y_pred, _ = collect_predictions(model, loader, device)
    return dataset, y_true, y_pred


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate cached latent attention classifier on val/test splits')
    parser.add_argument('--clean_latent_root', type=str, required=True)
    parser.add_argument('--aux_latent_root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--val_split', type=str, default='val')
    parser.add_argument('--test_split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--classifier_kind', type=str, choices=['auto', 'vector_attention', 'map_cnn', 'map_attention'], default='auto')
    parser.add_argument('--map_hidden_dim', type=int, default=128)
    parser.add_argument('--map_depth', type=int, default=4)
    parser.add_argument('--map_dropout', type=float, default=0.0)
    parser.add_argument('--path_contains', nargs='+', default=None)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--per_generator', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    clean_root = Path(args.clean_latent_root).resolve()
    aux_root = Path(args.aux_latent_root).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    checkpoint = torch.load(ckpt_path, map_location=device)

    val_dataset = LatentPairClassificationDataset(
        clean_latent_root=str(clean_root),
        aux_latent_root=str(aux_root),
        split=args.val_split,
        include_path_contains=args.path_contains,
    )
    ckpt_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
    classifier_kind_arg = checkpoint.get('classifier_kind') or ckpt_args.get('classifier_kind') or args.classifier_kind
    map_hidden_dim = int(ckpt_args.get('map_hidden_dim', args.map_hidden_dim))
    map_depth = int(ckpt_args.get('map_depth', args.map_depth))
    map_dropout = float(ckpt_args.get('map_dropout', args.map_dropout))
    model, classifier_kind = build_latent_pair_classifier(
        input_dim=val_dataset.feature_dim,
        is_spatial=val_dataset.is_spatial,
        classifier_kind=classifier_kind_arg,
        map_hidden_dim=map_hidden_dim,
        map_depth=map_depth,
        map_dropout=map_dropout,
    )
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)

    val_loader = build_loader(val_dataset, args.batch_size, args.num_workers, args.prefetch_factor, device)
    val_y_true, val_y_pred, _ = collect_predictions(model, val_loader, device)
    val_threshold = find_best_threshold(val_y_true, val_y_pred)
    results = {
        'checkpoint': str(ckpt_path),
        'clean_latent_root': str(clean_root),
        'aux_latent_root': str(aux_root),
        'val_split': args.val_split,
        'test_split': args.test_split,
        'classifier_kind': classifier_kind,
        'val_metrics': metrics_from_predictions(val_y_true, val_y_pred),
    }

    test_dataset = LatentPairClassificationDataset(
        clean_latent_root=str(clean_root),
        aux_latent_root=str(aux_root),
        split=args.test_split,
        include_path_contains=args.path_contains,
    )
    test_loader = build_loader(test_dataset, args.batch_size, args.num_workers, args.prefetch_factor, device)
    test_y_true, test_y_pred, _ = collect_predictions(model, test_loader, device)
    results['test_metrics'] = metrics_from_predictions(test_y_true, test_y_pred, fixed_threshold=val_threshold)

    if args.per_generator:
        generator_metrics = {}
        for generator in get_generators(clean_root, args.test_split):
            include_tokens = [generator]
            dataset, y_true, y_pred = evaluate_split(
                model,
                clean_root,
                aux_root,
                args.test_split,
                include_tokens,
                args.batch_size,
                args.num_workers,
                args.prefetch_factor,
                device,
            )
            generator_metrics[generator] = metrics_from_predictions(y_true, y_pred, fixed_threshold=val_threshold)
            generator_metrics[generator]['num_samples'] = len(dataset)
        results['per_generator_test_metrics'] = generator_metrics

    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
        print(f'[Saved] {output_path}')


if __name__ == '__main__':
    main()
