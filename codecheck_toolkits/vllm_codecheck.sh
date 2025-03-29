pip install -r requirements-lint.txt

cd ..
# yapf formats code automatically

MERGEBASE="$(git merge-base origin/master HEAD)"
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  yapf --in-place --recursive --parallel --exclude build/
fi

if [[ $? -ne 0 ]]; then
  echo "yapf run failed."
else
  echo "yapf run success."
fi

# codespell check
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  codespell --skip ./vllm_mindspore/ops/ascendc/*
fi
if [[ $? -ne 0 ]]; then
  echo "codespell check failed."
else
  echo "codespell check success."
fi

# ruff check
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  ruff check
fi
if [[ $? -ne 0 ]]; then
  echo "ruff check failed."
else
  echo "ruff check success."
fi

# isort fixed
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  isort
fi
if [[ $? -ne 0 ]]; then
  echo "isort fixed failed."
else
  echo "isort fixed success."
fi

# mypy check type
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  mypy --follow-imports skip --python-version 3.9 "$@"
fi
if [[ $? -ne 0 ]]; then
  echo "mypy check failed."
else
  echo "mypy check success."
fi

cd -
