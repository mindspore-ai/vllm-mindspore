# Copyright 2025 Huawei Technologies Co., Ltd
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

import inspect
import types
from functools import wraps
from mrt._mrt_api import DALangPy_


def _get_source(func):
    fn = inspect.unwrap(func.__func__ if isinstance(func, types.MethodType) else func)
    src_lines = inspect.getsourcelines(fn)
    lines, line_offset = src_lines
    src = ''.join(lines)
    return src


def jit(func=None, *, dump_compiler=False):
    def decorator(func):
        @wraps(func)
        def wrap_func(*args, **kwargs):
            dalang_py = DALangPy_.get_instance()
            if dalang_py is None:
                return func(args, kwargs)
            src = _get_source(func)
            dalang_py.compile(src, False, dump_compiler)
            return dalang_py(args)
        return wrap_func

    if func is None:
        return decorator
    return decorator(func)


def dag(func=None, *, dump_compiler=False):
    def decorator(func):
        @wraps(func)
        def wrap_func(*args, **kwargs):
            dalang_py = DALangPy_.get_instance()
            if dalang_py is None:
                return func(args, kwargs)
            src = _get_source(func)
            dalang_py.compile(src, True, dump_compiler)
            return dalang_py(args)
        return wrap_func

    if func is None:
        return decorator
    return decorator(func)