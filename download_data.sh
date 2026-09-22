#!/usr/bin/env bash
# Download Cricsheet ball-by-ball JSON zips (men's white-ball) for the rater.
# Reproducible: re-run any time; skips files already present and verified.
# Source: https://cricsheet.org/downloads/ (free, CC-BY-SA data)
set -euo pipefail

RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

# Competition slugs whose "<slug>_male_json.zip" returned HTTP 200 on 2026-09-21.
SLUGS=(
  t20s        # T20 internationals (men)      ~3,558 matches
  odis        # One-day internationals (men)  ~2,576 matches
  ipl         # Indian Premier League         ~1,243 matches
  bbl         # Big Bash League               ~662 matches
  psl         # Pakistan Super League         ~357 matches
  cpl         # Caribbean Premier League      ~442 matches
  bpl         # Bangladesh Premier League     ~469 matches
  mlc         # Major League Cricket          ~109 matches
  lpl         # Lanka Premier League          ~143 matches
  msl         # Mzansi Super League           ~56 matches
  ilt         # International League T20      ~134 matches
  npl         # Nepal Premier League          ~64 matches
  sma         # Syed Mushtaq Ali Trophy       ~695 matches
  etpl        # European T20 Premier League   ~29 matches
)

for slug in "${SLUGS[@]}"; do
  zip="${slug}_male_json.zip"
  if [ -f "$zip" ] && unzip -t -q "$zip" >/dev/null 2>&1; then
    echo "SKIP (ok): $zip"
    continue
  fi
  echo "GET: $zip"
  if ! curl -sS --retry 3 --max-time 600 -o "$zip" "https://cricsheet.org/downloads/$zip"; then
    echo "  WARN: download failed, skipping"; rm -f "$zip"; continue
  fi
  if ! unzip -t -q "$zip" >/dev/null 2>&1; then
    echo "  WARN: corrupt zip, skipping"; rm -f "$zip"; continue
  fi
  mkdir -p "$slug"
  unzip -q -o "$zip" -d "$slug"
  echo "  -> extracted to $slug/ ($(ls "$slug" | wc -l) matches)"
done

echo "DONE. Total match files: $(find . -name '*.json' | wc -l)"
