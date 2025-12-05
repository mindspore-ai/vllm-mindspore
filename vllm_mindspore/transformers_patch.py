# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import os
import sys
import types
import warnings

os.environ["USE_TORCH"] = "FALSE"
os.environ["USE_TF"] = "FALSE"


def _patch_processing_module():
    import mindone.transformers.models as mo_models
    import transformers.models as tf_models

    # Gather all modules in mindone.transformers.models
    for attr in dir(mo_models):
        if attr.startswith("_"):
            continue
        try:
            mo_submod = getattr(mo_models, attr)
            _ = getattr(tf_models, attr, None)
        except Exception:
            continue
        if not isinstance(mo_submod, types.ModuleType):
            continue

        # Get all "processing_"* or "*processing*"
        # or "image_processing_"* in mindone
        for sub_attr in dir(mo_submod):
            if ("processing" in sub_attr):
                try:
                    _ = getattr(mo_submod, sub_attr)
                except Exception:
                    continue

                # Try to get matching transformers module
                tf_modname = f"transformers.models.{attr}.{sub_attr}"
                mo_modname = f"mindone.transformers.models.{attr}.{sub_attr}"
                if tf_modname in sys.modules and mo_modname in sys.modules:
                    sys.modules[tf_modname] = sys.modules[mo_modname]
                else:
                    # Try to import if not already loaded
                    try:
                        _ = importlib.import_module(tf_modname)
                        _ = importlib.import_module(mo_modname)
                        sys.modules[tf_modname] = sys.modules[mo_modname]
                    except Exception:
                        continue


def patch_transformers():
    try:
        import mindone  # noqa: F401
    except ImportError:
        warnings.warn(
            "mindone.transformers not installed, "
            "skip patching transformers.",
            stacklevel=2)
        return

    import transformers
    from mindone.transformers import ProcessorMixin
    transformers.ProcessorMixin = ProcessorMixin
    transformers.processing_utils.ProcessorMixin = ProcessorMixin

    from mindone.transformers import AutoProcessor
    transformers.AutoProcessor = AutoProcessor
    transformers.models.auto.processing_auto.AutoProcessor = AutoProcessor

    from mindone.transformers import AutoImageProcessor
    transformers.AutoImageProcessor = AutoImageProcessor

    _patch_processing_module()
