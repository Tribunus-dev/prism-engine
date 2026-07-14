#!/bin/zsh
set -euo pipefail

log_dir="${1:?log directory is required}"
max_bytes=$((10 * 1024 * 1024))
keep=5
mkdir -p "$log_dir"

for log in prism-mcpd.log prism-mcpd.error.log; do
  path="${log_dir}/${log}"
  [[ -f "$path" ]] || continue
  size=$(/usr/bin/wc -c < "$path")
  (( size < max_bytes )) && continue
  for index in {5..1}; do
    previous="${path}.${index}.gz"
    next="${path}.$((index + 1)).gz"
    [[ -f "$previous" ]] && mv -f "$previous" "$next"
  done
  mv -f "$path" "${path}.1"
  gzip -f "${path}.1"
  rm -f "${path}.6.gz"
done
