# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the license.
# You may obtain a copy of the license at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# only check in CI
sed -i "18c\    args: [--diff]" .pre-commit-config.yaml  # yapf
sed -i "25c\    args: ['--exit-non-zero-on-fix']" .pre-commit-config.yaml  # ruff
sed -i "27i\    args: ['--diff']" .pre-commit-config.yaml  # ruff-format
sed -i "39i\    args: ['--check-only', '--diff']" .pre-commit-config.yaml  # isort

git add .pre-commit-config.yaml

pip install -r codecheck_toolkits/requirements-lint.txt

RUN_GATE=${1:-0}
if [ "$RUN_GATE" -eq 0 ]; then
  pre-commit run --from-ref origin/master --to-ref HEAD
else
  pre-commit run --all-files
fi
RET_FLAG=$?

exit ${RET_FLAG}
