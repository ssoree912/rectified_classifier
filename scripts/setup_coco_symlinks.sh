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
TRAIN_COCO_TXT="$TRAIN_DIFFUSION_ROOT/real_coco.txt"
VALID_COCO_TXT="$VALID_DIFFUSION_ROOT/real_coco.txt"

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
[[ -f "$TRAIN_COCO_TXT" ]] || { echo "missing file: $TRAIN_COCO_TXT" >&2; exit 1; }
[[ -f "$VALID_COCO_TXT" ]] || { echo "missing file: $VALID_COCO_TXT" >&2; exit 1; }

if [[ "$RESET" -eq 1 && -e "$ROOT" ]]; then
  rm -rf "$ROOT"
fi

mkdir -p "$ROOT/train/real" "$ROOT/valid" "$ROOT/test"

resolve_coco_src() {
  local raw="$1"
  local line
  line="$(printf '%s' "$raw" | tr -d '\r')"
  [[ -n "$line" ]] || return 1

  if [[ -f "$line" ]]; then
    printf '%s\n' "$line"
    return 0
  fi

  if [[ -f "$COCO_ROOT/$line" ]]; then
    printf '%s\n' "$COCO_ROOT/$line"
    return 0
  fi

  if [[ -f "$COCO_ROOT/train2017/$line" ]]; then
    printf '%s\n' "$COCO_ROOT/train2017/$line"
    return 0
  fi

  if [[ -f "$COCO_ROOT/val2017/$line" ]]; then
    printf '%s\n' "$COCO_ROOT/val2017/$line"
    return 0
  fi

  if [[ -f "$COCO_ROOT/test2017/$line" ]]; then
    printf '%s\n' "$COCO_ROOT/test2017/$line"
    return 0
  fi

  local base
  base="$(basename "$line")"
  if [[ -f "$COCO_ROOT/train2017/$base" ]]; then
    printf '%s\n' "$COCO_ROOT/train2017/$base"
    return 0
  fi
  if [[ -f "$COCO_ROOT/val2017/$base" ]]; then
    printf '%s\n' "$COCO_ROOT/val2017/$base"
    return 0
  fi
  if [[ -f "$COCO_ROOT/test2017/$base" ]]; then
    printf '%s\n' "$COCO_ROOT/test2017/$base"
    return 0
  fi

  return 1
}

populate_real_from_txt() {
  local txt_path="$1"
  local out_root="$2"
  mkdir -p "$out_root"

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    local src
    src="$(resolve_coco_src "$raw_line")" || {
      echo "warning: could not resolve COCO image from txt entry: $raw_line" >&2
      continue
    }

    local rel
    if [[ "$src" == "$COCO_ROOT/"* ]]; then
      rel="${src#$COCO_ROOT/}"
    else
      rel="$(basename "$src")"
    fi

    mkdir -p "$out_root/$(dirname "$rel")"
    ln -sfn "$src" "$out_root/$rel"
  done < "$txt_path"
}

# Train: txt-matched COCO real subset + COCO-compatible fake generators.
populate_real_from_txt "$TRAIN_COCO_TXT" "$ROOT/train/real/coco"
for name in "${TRAIN_FAKES[@]}"; do
  link_dir "$TRAIN_DIFFUSION_ROOT/$name" "$ROOT/train/$name"
done

# Valid: one shared txt-matched COCO real subset + multiple fake generators.
mkdir -p "$ROOT/valid/real"
populate_real_from_txt "$VALID_COCO_TXT" "$ROOT/valid/real/coco"
for name in "${VALID_FAKES[@]}"; do
  link_dir "$VALID_DIFFUSION_ROOT/$name" "$ROOT/valid/$name"
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
echo "Validation root:"
echo "  $ROOT/valid"
echo
echo "Test roots:"
for name in "${TEST_FAKES[@]}"; do
  echo "  $ROOT/test/$name"
done
