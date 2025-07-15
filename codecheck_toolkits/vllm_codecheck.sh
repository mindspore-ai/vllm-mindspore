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

sed -i "9c\    args: [--diff]" .pre-commit-config.yaml
sed -i "16c\    args: ['--exit-non-zero-on-fix']" .pre-commit-config.yaml
sed -i "18i\    args: ['--diff']" .pre-commit-config.yaml
sed -i "30c\    args: ['--check-only', '--diff']" .pre-commit-config.yaml

git add .pre-commit-config.yaml

pip install -r codecheck_toolkits/requirements-lint.txt

ln -s codecheck_toolkits/pyproject.toml pyproject.toml

pre-commit run --from-ref origin/develop --to-ref HEAD
RET_FLAG=$?

rm -f pyproject.toml

exit ${RET_FLAG}
