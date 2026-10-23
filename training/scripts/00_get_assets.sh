#!/usr/bin/env bash
# Fetch a stock Omnilingual CTC checkpoint + the char tokenizer, and the LibriSpeech subsets.
#
#   scripts/00_get_assets.sh [1b|300m]        # default 1b
#
# LibriSpeech tarballs are streamed straight into tar, so the archive never lands
# on disk: train-clean-360 alone is 23 GB. A subset counts as done only once its
# .done marker exists; rerunning the script skips finished parts and redoes an
# interrupted one from scratch. (scripts/fetch_parallel.sh downloads one tarball over
# several connections when the mirror caps each one.)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f local.env ]; then set -a; . ./local.env; set +a; fi

SIZE=${1:-1b}
case $SIZE in 1b) NAME=omniASR-CTC-1B ;; 300m) NAME=omniASR-CTC-300M ;; *) echo "size must be 1b or 300m"; exit 2 ;; esac
MODELS=${SOTA_MODELS_DIR:-$PWD/pretrained}
LIBRI_DIR=${LIBRI_DIR:-$PWD/corpora/librispeech}
MIRROR=${MIRROR:-https://www.openslr.org/resources/12}
SUBSETS=${SUBSETS:-"dev-clean test-clean test-other train-clean-100 train-clean-360"}

mkdir -p "$MODELS/$NAME" "$LIBRI_DIR"

fetch() {  # url dest
    [ -e "$2" ] && return
    echo "downloading $(basename "$2")"
    curl -fL --retry 10 --retry-delay 5 -C - -o "$2.part" "$1"
    mv "$2.part" "$2"
}
# a converted copy is all training needs; the original is only read once by the converter
[ -e "$MODELS/$NAME/$NAME.sota.pt" ] || fetch "https://dl.fbaipublicfiles.com/mms/$NAME.pt" "$MODELS/$NAME/$NAME.pt"
fetch "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model" "$MODELS/omniASR_tokenizer.model"

for s in $SUBSETS; do
    if [ -f "$LIBRI_DIR/.$s.done" ]; then
        echo "$s: already extracted"; continue
    fi
    echo "$s: streaming + extracting"
    rm -rf "$LIBRI_DIR/LibriSpeech/$s"
    curl -fL --retry 10 --retry-delay 5 "$MIRROR/$s.tar.gz" | tar -xz -C "$LIBRI_DIR"
    touch "$LIBRI_DIR/.$s.done"
done
echo "all assets ready"
