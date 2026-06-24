#!/usr/bin/env python3
"""Patch make_compiled_preamble.sh to handle paths with spaces.
Uses OLDIFS swap around the word-split: safe, minimal, persistent."""
import sys

script_path = sys.argv[1]
with open(script_path) as f:
    lines = f.readlines()

if any('IFS_PATCHED' in l for l in lines):
    sys.exit(0)

result = []
for line in lines:
    stripped = line.strip()

    # Fix 1: wrap HDRS_LIST assignment with IFS=newline
    if stripped == 'declare -a HDRS_LIST=($HDRS)':
        result.append('\tlocal OLDIFS=$IFS\n')
        result.append("\tIFS=$'\\n'\n")
        result.append(line)
        result.append('\tIFS=$OLDIFS\n')
        continue

    # Fix 2: with IFS=newline, each element is '. path', not separated.
    # Update array element access to extract depth/path from single element.
    if stripped == 'header="${HDRS_LIST[$i+1]#$SRC_DIR/}"':
        result.append('  line="${HDRS_LIST[$i]}"\n')
        result.append('  str_this="${line%% *}"\n')
        result.append('  header="${line#* }"\n')
        result.append('  header="${header#$SRC_DIR/}"\n')
        continue

    if stripped == 'str_this="${HDRS_LIST[$i]}"':
        continue  # merged into header block above

    if stripped == 'str_next="${HDRS_LIST[$i + 2]}"':
        result.append('  next_line="${HDRS_LIST[$i + 2]}"\n')
        result.append('  str_next="${next_line%% *}"\n')
        continue

    result.append(line)

# Add marker
output = ''.join(result)
output = output.replace(
    'make_compiled_preamble.sh',
    'make_compiled_preamble.sh (IFS_PATCHED)')

with open(script_path, 'w') as f:
    f.write(output)
print(f'Patched: {script_path}')
