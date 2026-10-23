#!/usr/bin/env bash
# Download one file as N parallel byte ranges (resumable), then optionally extract a .tar.gz.
# OpenSLR caps each connection at ~1.5 MB/s; several ranges use the whole line.
#
#   scripts/fetch_parallel.sh <url> <out_file> [parts] [extract_dir] [done_marker]
set -euo pipefail
URL=$1; OUT=$2; PARTS=${3:-4}; EXTRACT=${4:-}; MARKER=${5:-}

size=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
[ -n "$size" ] || { echo "no content-length for $URL"; exit 1; }
chunk=$(( (size + PARTS - 1) / PARTS ))
echo "$(date '+%F %T') $URL: $size bytes in $PARTS parts"

fetch_part() {
    local i=$1 start=$(( $1 * chunk )) end=$(( ($1 + 1) * chunk - 1 ))
    [ "$end" -ge "$size" ] && end=$(( size - 1 ))
    local want=$(( end - start + 1 )) f="$OUT.part$i"
    touch "$f"
    while :; do
        have=$(stat -c %s "$f")
        [ "$have" -ge "$want" ] && break
        curl -sfL --retry 20 --retry-delay 5 -r "$(( start + have ))-$end" "$URL" >> "$f" || sleep 10
    done
}
pids=()
for ((i = 0; i < PARTS; i++)); do fetch_part "$i" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done

parts=$(for ((i = 0; i < PARTS; i++)); do echo "$OUT.part$i"; done)
total=$(stat -c %s $parts | paste -sd+ | bc)
[ "$total" -eq "$size" ] || { echo "size mismatch: $total != $size"; exit 1; }
echo "$(date '+%F %T') downloaded $OUT ($PARTS parts)"
if [ -n "$EXTRACT" ]; then
    # stream the parts into tar: joining them first would need the archive's size twice
    cat $parts | tar -xz -C "$EXTRACT"
    rm -f $parts
    [ -n "$MARKER" ] && touch "$MARKER"
    echo "$(date '+%F %T') extracted into $EXTRACT"
else
    cat $parts > "$OUT" && rm -f $parts
fi
