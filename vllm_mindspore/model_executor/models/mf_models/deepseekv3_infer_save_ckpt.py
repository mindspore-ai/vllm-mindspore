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
# ============================================================================
"""Infer save ckpt by safetensor."""
import argparse
import os
from collections import OrderedDict

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore.communication.management import get_rank
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools.logger import logger
from mindformers import MindFormerConfig
from mindformers import build_context
from research.deepseek3.deepseekv3_infer_parallelism import DeepseekInferParallelism

from research.deepseek3.deepseek3_config import DeepseekV3Config
from research.deepseek3.deepseek3_model_infer import InferenceDeepseekV3ForCausalLM

# for example
# bash scripts/msrun_launcher.sh "python ./infer_save_ckpt_from_safetensor.py
# --config /path/to/predict_deepseek_r1_671b.yaml
# --save_ckpt_path /path/to/save_ckpt_path
# --load_checkpoint /path/to/safetensor_path " 4 8555 "output/deepseek_msrun_log" "False" 7200

def create_ptq():
    '''create_ptq'''
    from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
    from mindspore_gs.ptq import PTQ
    from mindspore_gs.common import BackendTarget
    from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, PrecisionRecovery, QuantGranularity
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                    act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS,
                    opname_blacklist=['lkv2kv', 'lm_head'], precision_recovery=PrecisionRecovery.NONE,
                    act_quant_granularity=QuantGranularity.PER_TENSOR,
                    weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    ffn_config = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                           act_quant_dtype=msdtype.int8,
                           outliers_suppression=OutliersSuppressionType.NONE,
                           precision_recovery=PrecisionRecovery.NONE,
                           act_quant_granularity=QuantGranularity.PER_TOKEN,
                           weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    ptq = PTQ(config=cfg, layer_policies=OrderedDict({r'.*\.feed_forward\..*': ffn_config}))
    ptq.decoder_layers.append(DeepseekV3DecodeLayer)
    return ptq


def main(config_path, load_checkpoint, save_ckpt_dir):
    # set model config
    config = MindFormerConfig(config_path)
    config.load_checkpoint = load_checkpoint

    build_context(config)
    build_parallel_config(config)
    model_config = config.model.model_config
    model_config.parallel_config = config.parallel_config
    model_config.moe_config = config.moe_config
    model_config = DeepseekV3Config(**model_config)

    # build model from config
    network = InferenceDeepseekV3ForCausalLM(model_config)

    is_quant = hasattr(config.model.model_config, "quantization_config")

    if is_quant:
        ptq = create_ptq()
        ptq.apply(network)
        ptq.convert(network)
        ptq.summary(network)
    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        model_parallelism = DeepseekInferParallelism(config, network, is_quant)
        model_parallelism.infer_convert_and_parallelism(config.load_checkpoint)

    rank_id = str(get_rank())
    os.makedirs(os.path.join(save_ckpt_dir, "rank_" + rank_id), exist_ok=True)

    save_ckpt_path = os.path.join(save_ckpt_dir, "rank_" + rank_id, "checkpoint_" + rank_id + ".ckpt")
    ms.save_checkpoint(network.parameters_dict(), save_ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='predict_llama2_7b.yaml', type=str,
                        help='model config file path.')
    parser.add_argument('--load_checkpoint', type=str,
                        help='load model checkpoint path or directory.')
    parser.add_argument('--save_ckpt_dir', type=str,
                        help='save ckpt path.')
    args = parser.parse_args()
    main(args.config_path, args.load_checkpoint, args.save_ckpt_dir)
