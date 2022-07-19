#!/bin/bash

# Only run tests in second-level directories that have been changed in the last commit.

# Fetch changed files.
mapfile -t dirs < <( git diff --dirstat=files,0 HEAD~1 | sed 's/^[ 0-9.]\+% //g')
declare -a tested_dirs=()
for dir in "${dirs[@]}"
do
  # Get path with second level only. This will be empty if the change happened at the first level.
  second_level_dir=$(echo "$dir" | awk -F/ '{print FS $2}')
  second_level_dir="${second_level_dir:1}"
  # Get path with first level/second level.
  full_second_level_dir=$(echo "$dir" | cut -d/ -f1-2)

  # Only run if change happened at second level, since first level-changes don't require the tests to be re-run.
  if [ ! -z "$second_level_dir" -a "$second_level_dir" != " " ]; then
    if [[ ! " ${tested_dirs[*]} " =~ " ${full_second_level_dir} " ]]; then
      tested_dirs+=($full_second_level_dir)
      if [ -e $full_second_level_dir/requirements.txt ]; then
        python -m pip install -r $full_second_level_dir/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
      fi
      python -m pytest -s $full_second_level_dir
    fi
  fi

done
