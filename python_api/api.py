#
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


import os
import inspect
import types
from functools import wraps
from ctypes import cdll

from _dalang import DALangPy_ 


def dag(func):
    @wraps(func)
    def wrap_func(*args, **kwargs):
        if DALangPy_.get_instance() is None:
            return func(args, kwargs)
        fn = inspect.unwrap(func.__func__ if isinstance(func, types.MethodType) else func)
        src_lines = inspect.getsourcelines(fn)
        lines, line_offset = src_lines
        src = ''.join(lines)
        DALangPy_.get_instance().compile(src)
        return DALangPy_.get_instance().run(callable)

    return wrap_func


@dag
def test_func():
    print('hello, world')
    return 0


test_func()