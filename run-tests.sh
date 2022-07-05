#!/bin/bash

# Only run tests in second-level directories that have been changed in the last commit.
mapfile -t dirs < <( git diff --dirstat=files,0 HEAD~1 | sed 's/^[ 0-9.]\+% //g')
declare -a tested_dirs=()
for dir in "${dirs[@]}"
do
  second_level_dir=$(echo "$dir" | awk -F/ '{print FS $2}')
  second_level_dir="${second_level_dir:1}"

  if [ ! -z "$second_level_dir" -a "$second_level_dir" != " " ]; then
    if [[ ! " ${tested_dirs[*]} " =~ " ${second_level_dir} " ]]; then
      tested_dirs+=($second_level_dir)
      python -m pytest -s $second_level_dir
    fi
  fi

done