#!/bin/bash

# Make sure Python interpreter is supplied.
if (( $# == 0 )); then
    >&2 echo "Python interpreter (e.g. 'python', 'python3.7') must be supplied."
    exit 1
fi
python_interpreter=$1

# Only run tests in second-level directories that have been changed in the last commit.
status=0

# Fetch changed files.
declare -a tested_dirs=()

# For complete run independent from Git changes: bash run-tests.sh $python_interpreter all
if [[ $2 == "all" ]]; then
  echo "Executing all tests"
  declare -a project_collections=("benchmarks" "experimental" "integrations" "pipelines" "tutorials")
  declare -a dirs=()
  for dir in "${project_collections[@]}"
  do
    # Make sure parsing `find` into array works on bash < 4.4 (which seems to be the case for our BuildKite AMIs).
    # 4.4+ is easier, see https://stackoverflow.com/a/23357277.
    second_level_dirs=()
    while IFS=  read -r -d $'\0'; do
        second_level_dirs+=("$REPLY")
    done < <(find ${dir} -mindepth 1 -maxdepth 1 -type d -print0)

    for second_level_dir in "${second_level_dirs[@]}"
    do
      dirs+=($second_level_dir)
    done
  done
else
  echo "Executing tests in changed directories"
  mapfile -t dirs < <( git diff --dirstat=files,0 HEAD~1 | sed 's/^[ 0-9.]\+% //g')
fi

echo "Running tests in:"
for dir in "${dirs[@]}"
do
  echo $dir
done

for dir in "${dirs[@]}"
do
  # Get path with second level only. This will be empty if the change happened at the first level.
  second_level_dir=$(echo "$dir" | awk -F/ '{print FS $2}')
  second_level_dir="${second_level_dir:1}"
  # Get path with first level/second level.
  full_second_level_dir=$(echo "$dir" | cut -d/ -f1-2)

  # Only run if change happened at second level, since first level-changes don't require the tests to be re-run.
  # If change happened at first level, $second_level_dir will be empty.
  if [ ! -z "$second_level_dir" -a "$second_level_dir" != " " ]; then
    if [[ ! " ${tested_dirs[*]} " =~ " ${full_second_level_dir} " ]]; then
      echo "Executing tests for $full_second_level_dir"

      tested_dirs+=($full_second_level_dir)
      if [ -e $full_second_level_dir/requirements.txt ]; then
        $python_interpreter -m pip -q install -r $full_second_level_dir/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
      fi

      # Ensure proper spaCy version is installed.
      spacy_version=$(
        $python_interpreter -c "import srsly; data = srsly.read_yaml('${full_second_level_dir}/project.yml'); print(data.get('spacy_version', ''))"
      )
      if [ ! -z "$spacy_version" ]; then
        $python_interpreter -m pip -q install "spacy${spacy_version}" --force-reinstall --no-warn-script-location
      fi

      $python_interpreter -m pytest -q -s $full_second_level_dir

      # Mark as failure if exit code isn't either 0 (success) or 5 (no tests found).
      if [[ $? != @(0|5) ]]; then
        status=1
      fi

      if [ -e $full_second_level_dir/requirements.txt ]; then
        $python_interpreter -m pip freeze --exclude torch cupy-cuda111 > installed.txt
        $python_interpreter -m pip -q uninstall -y -r installed.txt
        $python_interpreter -m pip -q install pytest spacy aiohttp --no-warn-script-location
        rm installed.txt
      fi
    fi
  fi
done

exit $status
