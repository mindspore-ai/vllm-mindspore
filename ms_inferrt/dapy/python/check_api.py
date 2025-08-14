# Copyright 2025 Zhang Qinghua
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dapy import jit
import argparse

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument('--dump', '-d', type=bool, default=False, required=False, help="if dump compiler information")
_args = _arg_parser.parse_args()


@jit(dump_compiler=_args.dump)
def run_check(x, y):
    print('hello world.\n')
    z = x * y
    z = z + x - y
    z = z / y
    return z

assert run_check(12, 6) == 13
print("The result is correct. 'dapy' module has been installed successfully.")