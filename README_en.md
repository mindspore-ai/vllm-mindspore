<h1 align="center">
vLLM MindSpore
</h1>

<p align="center">
| <a href="https://www.mindspore.cn/en/"><b>About MindSpore</b></a> | <a href="https://www.mindspore.cn/community/SIG"><b>vLLM MindSpore SIG</b></a> | <a href="https://gitee.com/mindspore/vllm-mindspore/issues"><b>Issue Feedback</b></a> |
</p>

<p align="center">
<a href="README_en.md"><b>English</b></a> | <a href="README.md"><b>中文</b></a>
</p>

---
*Latest News* 🔥

- [2025/06] Adaptation for vLLM [v0.8.3](https://github.com/vllm-project/vllm/releases/tag/v0.8.3), support for vLLM V1 architecture and the Qwen3 large model.
- [2025/04] Adaptation for vLLM [v0.7.3](https://github.com/vllm-project/vllm/releases/tag/v0.7.3), support Automatic Prefix Caching, Chunked Prefill, Multi-step Scheduling, and MTP. In collaboration with the openEuler community and Shanghai Jiao Tong University, we achieved full-stack open-source single-machine inference deployment for DeepSeek. You can read the detailed report [here](https://news.pku.edu.cn/xwzh/e13046c47d03471c8cebb950bd1f4598.htm).
- [2025/03] Adaptation for vLLM [v0.6.6.post1](https://github.com/vllm-project/vllm/releases/tag/v0.6.6.post1) supporting the deployment of inference services for large models such as DeepSeek-V3/R1 and Qwen2.5 based on MindSpore using `vllm.entrypoints`. In collaboration with the openEuler community and Peking University, we released a full-stack open-source DeepSeek inference solution. You can read the detailed report [here](https://news.pku.edu.cn/xwzh/e13046c47d03471c8cebb950bd1f4598.htm).
- [2025/02] The MindSpore community officially created the [mindspore/vllm-mindspore](https://gitee.com/mindspore/vllm-mindspore) repository, aiming to integrate MindSpore's large model inference capabilities into vLLM.

---

# Overview

vLLM MindSpore (`vllm-mindspore`) is a plugin brewed by the [MindSpore community](https://www.mindspore.cn/en), which aims to integrate MindSpore LLM inference capabilities into [vLLM](https://github.com/vllm-project/vllm). With vLLM MindSpore, technical strengths of Mindspore and vLLM will be organically combined to provide a full-stack open-source, high-performance, easy-to-use LLM inference solution.

vLLM MindSpore plugin aims to integrate Mindspore large models into vLLM and to enable deploying MindSpore-based LLM inference services. It follows the following design principles:

- Interface compatibility: support the native APIs and service deployment interfaces of vLLM to avoid adding new configuration files or interfaces, reducing user learning costs and ensuring ease of use.
- Minimal invasive modifications: minimize invasive modifications to the vLLM code to ensure system maintainability and evolvability.
- Component decoupling: minimize and standardize the coupling between MindSpore large model components and vLLM service components to facilitate the integration of various MindSpore large model suites.

On the basis of the above design principles, vLLM MindSpore adopts the system architecture shown in the figure below, and implements the docking between vLLM and Mindspore in categories of components:

- Service components: vLLM MindSpore maps PyTorch API calls in service components including LLMEngine and Scheduler to MindSpore capabilities, inheriting support for service functions like Continuous Batching and PagedAttention.
- Model components: vLLM MindSpore registers or replaces model components including models, network layers, and custom operators, and integrates MindSpore Transformers, MindSpore One, and other MindSpore large model suites, as well as custom large models, into vLLM.

<div align="center">
  <img src="docs/arch.png" alt="Description" width="800" />
</div>

vLLM MindSpore uses the plugin mechanism recommended by the vLLM community to realize capability registration. In the future, we expect to follow principles described in [[RPC] Multi-framework support for vllm](https://gitee.com/mindspore/vllm-mindspore/issues/IBTNRG).

# Prerequisites

- Hardware：Atlas 800I A2 Inference series, or Atlas 800T A2 Training series, with necessary drivers installed and access to the Internet.
- Operating System: openEuler or Ubuntu Linux.
- Software:
  - Python >= 3.9, < 3.12
  - CANN >= 8.0.0.beta1
  - MindSpore
  - vLLM

Note: Please refer to [Version Compatibility](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/installation/installation.md) for more details about version compatibility information.

# Getting Started

Please refer to [Quick Start](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/quick_start/quick_start.md) and [Installation](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/getting_started/installation/installation.md) for more details.

# Contributing

Please read [CONTRIBUTING](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/developer_guide/contributing.md) for details on setting up development environments, testing functions, and submitting PR.

We welcome and value any form of contribution and cooperation. Please use [Issue](https://gitee.com/mindspore/vllm-mindspore/issues) to inform us of any bugs you encounter, or to submit your feature requests, improvement suggestions, and technical solutions.

# SIG

- Welcome to join vLLM MindSpore SIG to participate in the co-construction of open-source projects and industrial cooperation: [https://www.mindspore.cn/community/SIG](https://www.mindspore.cn/community/SIG)
- SIG meetings, every other Wednesday or Thursday afternoon, 16:30 - 17:30 (UTC+8,   [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))
