# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/models/adapters.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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

from collections.abc import Iterable
from typing import Any, cast

import mindspore as ms
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces_base import VllmModelForPooling


def _create_pooling_model_cls(orig_cls):
    # Lazy import
    from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                                  WeightsMapper)

    class ModelForPooling(orig_cls, VllmModelForPooling):

        is_pooling_model = True

        def __init__(
            self,
            *,
            vllm_config: "VllmConfig",
            prefix: str = "",
            **kwargs: Any,
        ) -> None:
            super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

            self.vllm_config = vllm_config

            # These are not used in pooling models
            for attr in ("lm_head", "logits_processor"):
                if hasattr(self, attr):
                    delattr(self, attr)
                    self.modules_dict.pop(attr, None)

            # If the model already defines a pooler instance, don't overwrite it
            if not getattr(self, "pooler", None):
                self._init_pooler(vllm_config, prefix=prefix)

        def _init_pooler(self, vllm_config: "VllmConfig", prefix: str = ""):
            raise NotImplementedError

        def load_weights(self, weights: Iterable[tuple[str, ms.Tensor]]):
            # TODO: Support uninitialized params tracking

            # We have deleted this attribute, so don't load it
            weights = ((name, data) for name, data in weights
                       if not name.startswith("lm_head."))

            # Using WeightsMapper to rename weights for fitting pooling model
            mapper = WeightsMapper(
                orig_to_new_prefix={
                    "layers.": "model.layers.",
                    "embed_tokens.": "model.embed_tokens.",
                    "norm.": "model.norm."
                })
            weights = mapper.apply(weights)

            # For most other models
            if hasattr(orig_cls, "load_weights"):
                return orig_cls.load_weights(self, weights)
            # Fallback
            else:
                loader = AutoWeightsLoader(self)
                return loader.load_weights(weights)

    return ModelForPooling  # type: ignore


def load_weights_using_from_2_way_softmax(model,
                                          weights: Iterable[tuple[str,
                                                                  ms.Tensor]]):
    # refer to https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
    # vllm-mindspore begin: use mindspore implementations
    from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead)

    # vllm-mindspore end.

    model_config = model.vllm_config.model_config

    tokens = getattr(model.config, "classifier_from_token", [])
    tokens = cast(list[int], tokens)
    assert len(tokens) == 2

    if model.config.tie_word_embeddings:
        model.lm_head = model.model.embed_tokens
    else:
        quant_config = model.vllm_config.quant_config
        # vllm-mindspore begin: update modules_dict
        # and add params_dtype in lm_head
        model.lm_head = ParallelLMHead(
            model.config.vocab_size,
            model.config.hidden_size,
            quant_config=quant_config,
            params_dtype=model.vllm_config.model_config.dtype)
        model.modules_dict.update({"lm_head": model.lm_head})
        # vllm-mindspore end.

    # vllm-mindspore begin: load weights using mindspore method
    params_dict = model.get_params_dict()
    loaded_weights = model.model.load_weights(weights, params_dict)
    # vllm-mindspore end.

    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(model_config.tokenizer,
                              revision=model_config.tokenizer_revision,
                              tokenizer_mode=model_config.tokenizer_mode,
                              trust_remote_code=model_config.trust_remote_code)

    false_id = tokenizer.convert_tokens_to_ids(tokens[0])
    true_id = tokenizer.convert_tokens_to_ids(tokens[1])
    score_weight = model.lm_head.weight.data[[true_id]].to(
        ms.float32) - model.lm_head.weight.data[[false_id]].to(ms.float32)

    param = model.score.weight
    param.set_data(score_weight)

    del model.lm_head
    # vllm-mindspore begin: remove lm_head from modules_dict
    model.modules_dict.pop("lm_head", None)
    # vllm-mindspore end.

    loaded_weights.add("score.weight")
    loaded_weights.discard("lm_head.weight")
    return loaded_weights


from vllm.model_executor.models.adapters import (  # noqa: E402
    SEQ_CLS_LOAD_METHODS)

# vllm-mindspore begin: substitute load method
SEQ_CLS_LOAD_METHODS["from_2_way_softmax"] = \
    load_weights_using_from_2_way_softmax
# vllm-mindspore end.