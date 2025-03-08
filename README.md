# vllm-mindspore

## Overview

The `vllm-mindspore`is a integration for running vLLM on the MindSpore framework.

This  is the recommended solution for supporting the MindSpore  within the vLLM community. It provides deep integration with the MindSpore framework, offering efficient computation and optimization support for vLLM, enabling seamless operation on MindSpore.

By using the `vllm-mindspore`, popular open-source models, can run seamlessly for training and inference on the MindSpore framework.

---

## Prerequisites

- Hardware: Atlas A2/A3
- Software:
    - Python >= 3.9
    - CANN >= 8.0.0
    - MindSpore >=2.5.0

---

## Getting Started

### Installation

#### Installation from source code

Install from source code. [Wiki Installation.](https://gitee.com/mindspore/vllm-mindspore/wikis/Getting%20Started/Installation)

#### Set up using Docker

##### Pre-built images

```shell
docker pull hub.oepkgs.net/oedeploy/openeuler/aarch64/mindspore:v1.0
```

##### Build image from source

```shell
docker build --network=host .
```

### Inference and Serving

#### Offline Inference

You can run vllm_mindspore in your own code on a list of prompts.

```bash
export ASCEND_TOTAL_MEMORY_GB=64 # Based on the ascend device.
```

```python

import vllm_mindspore # Add this line on the top of script.

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "I am",
    "Today is",
    "What is"
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=0.95)

# Create an LLM.
llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=8)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```

#### Serving（OpenAI-Compatible）

You can start the server via the vllm_mindspore command:

`python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server --model "Qwen/Qwen2.5-32B-Instruct" --tensor_parallel_size=8`

To call the server, you can use `curl` or any other HTTP client.

```shell

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "prompt": "MindSpore is",
    "max_tokens": 120,
    "temperature": 0
  }'

```

## Contributing

We welcome and value any contributions and collaborations:

- Please feel free comments about your usage of vllm_mindspore.
- Please let us know if you encounter a bug by filing an issue.

## License

Apache License 2.0, as found in the [LICENSE](https://gitee.com/mindspore/vllm_mindspore/blob/master/LICENSE) file.
