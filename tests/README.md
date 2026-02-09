<h1 align="center">
门禁DT用例规范
</h1>

# 一、文档概述

## 1.1 目的

vLLM-MindSpore插件将MindSpore大模型接入vLLM，继承vLLM服务化部署功能，随着各组件版本功能演进，功能和性能的正确性也成为非常重要的事情.因此考虑从开发者测试角度, 构建分层分级的测试能力，便于版本快速迭代和稳定交付。本文档为开源门禁DT用例建立统一的用例规范，确保功能设计符合用户实际需求、场景覆盖全面；同时提供清晰的标准，降低社区成员参与门槛，保障代码质量、一致性和项目迭代效率，共建可靠、易用、可扩展的门禁解决方案。

* 加速开发迭代效率：分层分级自动化测试即时反馈问题，避免在后续阶段花费大量时间定位解决；支撑各组件配套版本快速升级。
* 提升代码质量，减少交付质量风险: 自动化测试和相对完备测试用例覆盖，保障出口版本可靠。

## 1.2 适用范围

* 适用对象：所有参与项目开发、测试、文档编写、需求反馈的社区成员（贡献者）、维护者（Maintainer）。
* 覆盖内容：门禁核心功能用例定义、用例描述规范。

## 1.3 术语定义

* 门禁：代码门禁是一种有效保证产品代码满足要求的机制，是指开发人员提交PR后，需要触发完成对代码进行的检查，以保证代码满足规范标准，门禁检查的内容包括编码规范检查和测试用例检查等。
* DT（Developer Test）：即开发者测试, 由编写代码的工程师主导实施的测试活动；
* 用例：描述特定场景下开发者提供的测试代码段，提供执行预期结果，用于明确功能需求是否满足。
* PR（Pull Request）：贡献者向项目主分支提交的代码 / 文档修改请求。
* Issue：用于提交 Bug 反馈、功能需求、用例建议的社区沟通渠道。
* 简层用例：简化层数的权重，用于快速验证，如4层deepseek。

# 二、vLLM-MindSpore Plugin的门禁架构以及用例级别介绍

## 2.1 目录结构

主要目录级别如下所示，可根据实际调整新增:

```text
tests 
   ├── __init__.py
   ├── ut/                                 - Unit Test 单元测试用例目录；
   |   ├── ops/                            - 自定义算子用例；
   |   ├── samplers/                       - 采样器相关单元测试；
   |   ├── ...
   |   └── register_parallel_tests.json    - Unit Test 多实例并发执行注册列表            
   ├── st/                                 - System Test 系统测试用例目录。
   |   ├── models/                         - 在线模型相关用例；
   |   |  ├── deepseek/                    - DeepSeek在线模型相关用例；
   |   |  |   ├── offline/                 - 离线服务化用例
   |   |  |   |   └── test_ds_xx           - DeepSeek简层门禁用例；
   |   |  |   └── online/                  - 在线服务化用例  
   |   |  |       ├── test_ds_xx           - DeepSeek简层门禁用例；
   |   |  |       ├── pref_test/           - DeepSeek整网性能用例（含量化）；
   |   |  |       ├── acc_test/            - DeepSeek整网数据集精度用例（含量化）；
   |   |  |       └── endur_test/          - DeepSeek整网长稳用例；
   |   |  ├── qwen3/                       - Qwen3在线模型相关用例；
   |   |  └── ... 
   |   ├── lora/                           - multilora特性离线相关用例；
   |   ├── quantization/                   - 量化特性离线相关用例；
   |   ├── ep/                             - EP在线模型相关用例；
   |   |   ├── offline/                    - 离线服务化用例
   |   |   └── online/                     - 在线服务化用例  
   |   ├── ...
   |   └── register_parallel_tests.json    - System Test 多实例并发执行注册列表
   ├── utils                         
   |   ├── common_utils.py           - 通用模块定义
   |   ├── model_info.yaml           - 模型权重注册列表
   |   └── env_var_manager.py        - 环境变量相关模块定义
   └── test_cases_parallel.py        - 用于在线生成多实例并发执行用例组
```

## 2.2 用例常用配置

| 用例级别 | 触发时机                     | 环境资源      | 执行频率       | 用例内容 | 重点看护内容       | 当前状态(25.09)        |
| ---------- | ------------------------------ | --------------- | ---------------- | ---------- | -------------------- | ------------------------ |
| level0   | vLLM-MindSpore代码仓CI       | 公共门禁机器  | PR合入         | ut/st    | 基础功能           | 具备基础功能，24个用例 |
| level1   | vLLM-MindSpore代码仓每日构建 | 公共门禁机器  | 每天至少1次    | ut/st    | 重点特性           | 工程搭建中，8个用例    |
| level2   | 按需手动触发                 | ModelArts环境 | 特性开发合入前 | st       | 特性交互，精度看护 | 待构建                 |
| level3   | 按需手动触发                 | ModelArts环境 | 每周           | st       | 模型性能看护       | 待构建                 |
| level4   | 按需手动触发                 | ModelArts环境 | 按需           | st       | 长稳               | 待构建                 |

# 三、如何开发vLLM-MindSpore Plugin的门禁用例

## 3.1 基本规范

新增用例请遵循下述基本规范

```python
...test_{test_file_name}.py                       # 测试用例文件以"test_"前缀开头，尽量体现测试模块内容。文件内支持同时实现单卡/多卡等多个用例。

from tests.utils.common_utils import (teardown_function,     # [必选]pytest执行测试用例完成后会自动调用，主要用于清除服务化进程残留和释放资源
                                      setup_function)        # [必选]pytest执行测试用例前会自动调用，主要指定device卡和通信端口号


@patch.dict(os.environ, env_vars)                 # [可选]env_vars为自定义环境变量列表，如有可配置
@pytest.mark.{level_mark}                         # [必选]指定用例级别，当前支持level0~level4
@pytest.mark.{platform_mark}                      # [单卡/8卡场景须配置;其余卡数不配置,参考3.3章节]指定执行硬件，当前支持platform_arm_ascend910b_training和platform_ascend310p
@pytest.mark.{env_card_mark}                      # [单卡/8卡场景须配置;其余卡数不配置,参考3.3章节]指定执行卡数，单卡:env_onecard, 8卡:allcards    
@pytest.mark.parametrize("batch_size", [1, 2])    # [可选]测试用例参数化输入,如有可配置
def test_{test_function_name}(batch_size: int):   # 用例函数名以"test_"前缀开头，尽量体现测试场景
    """
    Test Summary:   # [必填]
        描述测试用例的测试场景，推荐使用英文
    Expected Result: # [必填]
        预期执行结果
    Model Info:     # [可选]
        如果用例涉及权重加载或模型配置
    """
    测试用例实现代码
```

标记为必选的标签必须添加，否则门禁系统无法正常识别执行用例。

 **注意:** 针对需要调用vllm原生接口的用例，需要提前导入`vllm_mindspore`，但由于`VLLM_MS_MODEL_BACKEND`等环境变量需要在导入`vllm_mindspore`前生效，推荐在用例内导入保证环境变量的有效性。如下示例:

```python
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
}

@patch.dict(os.environ, env_vars)
@pytest.mark.level0
def test_deepseek_r1():
    import vllm_mindspore
    from vllm import LLM, SamplingParams
```

## 3.2 基础示例(单卡/8卡场景)

* 单卡910B用例示例(推荐)

```python
@pytest.mark.level0 
@pytest.mark.platform_arm_ascend910b_training  
@pytest.mark.env_onecard
def test_qwen3_8B_enforce_eager():
    """
    Test Summary: 
        Test qwen3 8B using enforce_eager.
    Expected Result: 
        Running successfully, reasoning results are normal
    Model Info: 
        Qwen3-8B
    """
    run_vllm_qwen3_8b(enforce_eager=True)
```

* 8卡用例示例
部分模型规模较大/并行特性场景可添加8卡用例

```python
@patch.dict(os.environ, env_vars)
@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
def test_deepseek_r1_dp4_tp2_ep4():
    """
    Test Summary: 
        test case deepseek r1 w8a8 dp4 tp2 ep4
    Expected Result: 
        Running successfully, the first three tokens in the return result
    Model Info: 
        DeepSeek-R1-W8A8
    """
    import vllm_mindspore

    dp_size = 4
    tp_size = 2
    ep_size = 4
    # Sample prompts.
    prompts = [common_ds_prompt] * 4
    expect_list = [common_ds_expect_result] * 4
    exec_model_with_dp(dp_size, tp_size, ep_size, prompts, expect_list,
                       ds_model_path, quant_type)
```

## 3.3 并行用例示例(非单卡/8卡场景)

为了最大化门禁资源利用率，我们支持将多个用例组合并行执行，如同时执行1个4卡用例和两个2卡用例，避免资源空等。单卡用例门禁工程能力支持较好会自动并行执行，其余卡数用例需要遵循以下步骤:

 **Step1:** 实现用例，和单卡/8卡用例的区别在于不添加platform_mark和env_card_mark标记

```python
@pytest.mark.level0 
def test_qwen3_8B_online():
    """
    Test Summary: 
        Test qwen3 8B online.
    Expected Result: 
        Running successfully, reasoning results are normal
    Model Info: 
        Qwen3-8B
    """
    run_vllm_qwen3_8b()
```

 **Step2:** 注册并行配置，位于文件路经 tests/(st | ut)/register_parallel_tests.json，在对应的平台模块中注册。当前支持`registered_910b_tests`和`registered_310p_tests`。

```text
注册模板:
    {
      "test_node_id": String in {test_file_path}::{test_function_name} format,
      "card_num": Number of occupied cards
    },

```

注意:

1. `test_file_path` 应填写相对路经

2. `test_function_name` 对应的测试用例需要通过`@pytest.mark.{level_marks}`提前指定执行级别

Unit Test 注册示例如下:

```json
{
  "registered_910b_tests": [
    {
      "test_node_id": "ut/sampling/test_vllm_sampling.py::tst_vllm_sampling_n_logprobs",
      "card_num": 2
    }
  ],
  "registered_310p_tests": []
}
```

System Test 注册示例如下:

```json
{
  "registered_910b_tests": [
    {
      "test_node_id": "st/models/qwen2_5_vl/test_vllm_qwen2_5_vl_7b_v1.py::test_qwen2_5_vl_7b_v1",
      "card_num": 2
    },
    {
      "test_node_id": "st/distributed/test_shm_broadcast.py::test_shm_broadcast",
      "card_num": 4
    }
  ],
  "registered_310p_tests": [
    {
      "test_node_id": "st/models/qwen3/test_vllm_mf_qwen3_8b.py::test_mf_qwen3_v1_310p",
      "card_num": 2
    }
  ]
}

```

## 3.4 多机用例示例

暂不支持多机用例，待支持后补充。

## 3.5 权重上传与管理

### 3.5.1 权重上传

门禁使用的权重归档在特定服务器上，如果你的用例涉及新模型的权重，请通过PR或Issues联系Maintainer协助上传，您需要提供模型的开源权重下载连接。

同时在tests/utils/model_info.yaml中注册, 新增权重必须说明开源权重下载路经以及修改点。

```yaml
# 注册模板
{model_name}:
  description: {Describe the information and sources of weights}
  archive_addr: "https://tools.mindspore.cn/dataset/workspace/mindspore_dataset/weight/{model_name}"

# 例如

Llama-3.1-8B-Instruct:
  description: "Llama-3.1-8B, HF default configuration, source from https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
  archive_addr: "https://tools.mindspore.cn/dataset/workspace/mindspore_dataset/weight/Llama-3.1-8B-Instruct"
```

### 3.5.2 权重管理与使用

门禁环境现有权重可以在tests/utils/model_info.yaml查看已注册列表，如果已有相关模型权重可复用。
调用方式:

```text
from tests.utils.common_utils import MODEL_PATH

model_path = MODEL_PATH[{model_name}]
```

所有注册的权重都可以在权重归档地址[https://tools.mindspore.cn/dataset/workspace/mindspore_dataset/weight/](https://tools.mindspore.cn/dataset/workspace/mindspore_dataset/weight/)下载得到

## 3.6 确定性计算

针对精度验证场景，需要配置确定性计算环境变量，保证结果无随机性:

```python
env_vars = {
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}

@patch.dict(os.environ, env_vars)
```

## 3.7 本地环境用例执行

Case1: 本地全量用例执行

```shell

pytest tests -v --forked

```

注:依赖安装pytest-forked，`--forked`可保证每次用例执行环境变量正确初始化，不受前置用例影响。

Case2: 本地执行单卡用例

```shell
export DEVICE_ID=6    # 模拟在门禁场景下6卡执行该用例
pytest test_case_file.py::test_case
```

Case3: 本地执行多卡并行用例

```shell
pytest -sv tests/test_cases_parallel.py
```

注:开发者可以通过`-sv`打印日志信息查看是否正确执行对应用例，也可以通过注释`test_cases_parallel.py`中的UT/ST检索代码来控制是否执行对应逻辑，如下所示。

```python
retrieve_tests_from_path(ut_abs_path)
retrieve_tests_from_path(st_abs_path)
```

## 3.8 门禁报错定位分析

提交PR后会自动触发门禁CI，或通过评论`/retest`主动触发。门禁执行完成后会打上`ci-pipeline-passed`标签，否则会打上`ci-pipeline-failed`并反馈错误信息。
错误信息查看与分析流程:

 **Step1:** 查看报错信息，点击对应Details编号可查看具体报错信息。

![CI报错示例](https://foruda.gitee.com/images/1762998938604180842/f30932b7_7725520.png "屏幕截图")

 **Step2:** 点击门禁执行结果中对应失败的图标,会在屏幕下方呈现对应的详细报错日志

![门禁执行结果](https://foruda.gitee.com/images/1762999066849577106/351683ca_7725520.png "屏幕截图")

如果日志信息被折叠,可点击右上角图标查看全文或下载日志

![错误日志信息](https://foruda.gitee.com/images/1762999206560269222/901aa595_7725520.png "屏幕截图")

 **Step3:** 根据日志信息中的详细报错解决问题并重新提交
用例执行结果:

![用例执行结果](https://foruda.gitee.com/images/1762999311401431148/3685f0d8_7725520.png "屏幕截图")

报错信息:

![报错信息](https://foruda.gitee.com/images/1762999402368913470/c28eee3a_7725520.png "屏幕截图")

# 四、DT用例评审机制

新增/删除用例或调整用例级别需得到Maintainer同意

# 五、代码覆盖率统计

待补充...
