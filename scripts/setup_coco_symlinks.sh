#!/usr/bin/env bash
set -euo pipefail

RESET=0
if [[ "${1:-}" == "--reset" ]]; then
  RESET=1
  shift
fi

ROOT="${1:-/workspace/data/d3_data}"
COCO_ROOT="${COCO_ROOT:-/workspace/data/coco}"
DIFFUSION_ROOT="${DIFFUSION_ROOT:-/workspace/data/diffusion}"

TRAIN_DIFFUSION_ROOT="$DIFFUSION_ROOT/latent_diffusion_trainingset/train"
VALID_DIFFUSION_ROOT="$DIFFUSION_ROOT/latent_diffusion_trainingset/valid"
TEST_DIFFUSION_ROOT="$DIFFUSION_ROOT/TestSet"

TRAIN_FAKES=(
  latent_diffusion_text2img_set0
  latent_diffusion_text2img_set1
  latent_diffusion_text2img_set2
)

VALID_FAKES=(
  latent_diffusion_text2img_set0
  latent_diffusion_text2img_set1
  latent_diffusion_text2img_set2
)

TEST_FAKES=(
  dalle-mini_valid
  dalle_2
  glide_text2img_valid
  latent-diffusion_text2img_valid
  stable_diffusion_256
  taming-transformers_segm2image_valid
)

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "missing directory: $path" >&2
    exit 1
  fi
}

link_dir() {
  local src="$1"
  local dst="$2"
  require_dir "$src"
  mkdir -p "$(dirname "$dst")"
  ln -sfn "$src" "$dst"
}

require_dir "$COCO_ROOT/train2017"
require_dir "$COCO_ROOT/val2017"
require_dir "$COCO_ROOT/test2017"
require_dir "$TRAIN_DIFFUSION_ROOT"
require_dir "$VALID_DIFFUSION_ROOT"
require_dir "$TEST_DIFFUSION_ROOT"

if [[ "$RESET" -eq 1 && -e "$ROOT" ]]; then
  rm -rf "$ROOT"
fi

mkdir -p "$ROOT/train/real" "$ROOT/valid" "$ROOT/test"

# Train: one shared COCO real root + COCO-compatible fake generators.
link_dir "$COCO_ROOT/train2017" "$ROOT/train/real/coco"
for name in "${TRAIN_FAKES[@]}"; do
  link_dir "$TRAIN_DIFFUSION_ROOT/$name" "$ROOT/train/$name"
done

# Valid: one folder per generator so each root stays real/fake binary.
for name in "${VALID_FAKES[@]}"; do
  link_dir "$COCO_ROOT/val2017" "$ROOT/valid/$name/real"
  link_dir "$VALID_DIFFUSION_ROOT/$name" "$ROOT/valid/$name/fake"
done

# Test: one folder per generator so validate_for_robustness.py can read it directly.
for name in "${TEST_FAKES[@]}"; do
  link_dir "$COCO_ROOT/test2017" "$ROOT/test/$name/real"
  link_dir "$TEST_DIFFUSION_ROOT/$name" "$ROOT/test/$name/fake"
done

echo "Created COCO-compatible symlink layout under: $ROOT"
echo
echo "Train root:"
echo "  $ROOT/train"
echo
echo "Validation roots:"
for name in "${VALID_FAKES[@]}"; do
  echo "  $ROOT/valid/$name"
done
echo
echo "Test roots:"
for name in "${TEST_FAKES[@]}"; do
  echo "  $ROOT/test/$name"
done
