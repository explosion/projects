#!/usr/bin/env bash

# Only run tests in second-level directories that have been changed in the last commit.

# Fetch changed files.
function get_dirs() {
  git diff --dirstat=files,0 HEAD~1 | sed 's/^[ 0-9.]\+% //g' | cut -f 1-2 -d/ | grep / | sort -u
}

exit_code=0
get_dirs | while read dir
do
  if [ -e $dir/requirements.txt ]; then
    python -m pip install -q -r $dir/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  fi
  python -m pytest -s $dir
  # exit code 5 means no tests were found
  if [[ $? != 0 || $? != 5 ]]; then
    exit_code=1
  fi
done
exit $exit_code
