#!/bin/bash

# Only run tests in top-level directories that have been changed in the last commit.
mapfile -t dirs < <( git diff --dirstat=files,0 HEAD~1 | sed 's/^[ 0-9.]\+% //g')
declare -a tested_dirs=()
for dir in "${dirs[@]}"
do
  top_level_dir=$(echo "$dir" | awk -F/ '{print FS $1}')
  top_level_dir="${top_level_dir:1}"
  if [[ ! " ${tested_dirs[*]} " =~ " ${top_level_dir} " ]]; then
    tested_dirs+=($top_level_dir)
    python -m pytest -s $top_level_dir
  fi
done