#!/usr/bin/env bash
set -euo pipefail

if ! command -v pdal >/dev/null 2>&1; then
  printf 'Error: pdal is not installed or not on PATH.\n' >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  printf 'Usage: %s <scene-file> [<scene-file> ...]\n' "$0" >&2
  printf 'Prints PDAL summary and schema information for LAS/LAZ/COPC files.\n' >&2
  exit 1
fi

for path in "$@"; do
  if [ ! -f "$path" ]; then
    printf 'Skipping missing file: %s\n' "$path" >&2
    continue
  fi

  printf '=== %s ===\n' "$path"
  pdal info --summary "$path"
  printf '\n'
  pdal info --schema "$path"
  printf '\n'
done
