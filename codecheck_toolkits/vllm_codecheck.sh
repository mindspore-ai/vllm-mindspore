pip install -r codecheck_toolkits/requirements-lint.txt

ln -s codecheck_toolkits/pyproject.toml pyproject.toml

RET_FLAG=0

# yapf check

MERGEBASE="$(git merge-base origin/develop HEAD)"
if ! git diff --cached --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --cached --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  yapf --diff --recursive --parallel --exclude tests/
fi

if [[ $? -ne 0 ]]; then
  echo "yapf check failed."
  RET_FLAG=1
else
  echo "yapf check success."
fi

# codespell check
if ! git diff --cached --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --cached --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
  codespell --skip "./vllm_mindspore/ops/ascendc/*"
fi
if [[ $? -ne 0 ]]; then
  echo "codespell check failed."
  RET_FLAG=1
else
  echo "codespell check success."
fi

# ruff check
if ! git diff --cached --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  echo "ruff check is running..."
  git diff --cached --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' |  xargs \
  ruff check --force-exclude
fi
if [[ $? -ne 0 ]]; then
  echo "ruff check failed."
  RET_FLAG=1
else
  echo "ruff check success."
fi

# isort check
if ! git diff --cached --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --cached --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
  isort --check-only --diff
fi
if [[ $? -ne 0 ]]; then
  echo "isort check failed."
  RET_FLAG=1
else
  echo "isort check success."
fi

# mypy check type

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if ! git diff --cached --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --cached --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
  mypy --follow-imports skip --python-version "${PYTHON_VERSION}" "$@"
fi
if [[ $? -ne 0 ]]; then
  echo "mypy check failed."
  RET_FLAG=1
else
  echo "mypy check success."
fi

rm -f pyproject.toml

exit ${RET_FLAG}
