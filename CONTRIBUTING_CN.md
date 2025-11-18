<h1 align="center">
vLLM-MindSpore Plugin 开源社区贡献指南​
</h1>

欢迎加入vLLM-MindSpore插件开源社区！无论是提交代码、修复 Bug、优化文档，还是提出功能建议，你的每一份贡献都将帮助项目变得更好。本指南将为你提供完整的贡献流程、规范要求及支持渠道，确保贡献过程顺畅高效。​
## 一、贡献前准备​
### 1. 贡献者许可协议

- vLLM-MindSpore插件从属于MindSpore开源社区，向社区提交代码之前，您需要签署《贡献者许可协议（CLA）》。个人贡献者请参见[ICLA在线文件](https://www.mindspore.cn/icla)。

### 2. 了解项目​

- 仔细阅读项目根目录下的[README.md](https://gitee.com/mindspore/vllm-mindspore/blob/master/README.md)，明确项目定位、核心功能及技术栈。​

- 查看[Release Notes](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/RELEASE.html)和[里程碑](https://gitee.com/mindspore/vllm-mindspore/milestones)，了解项目历史更新记录和当前开发方向。​

### 3. 环境搭建​

- 按照[安装指南](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/getting_started/installation/installation.html)中的安装步骤，完成本地开发环境搭建。​

- 执行项目的本地测试命令（代码仓库根目录下执行 `pytest tests` ，需用户修改至本地权重路经），确保基础环境正常运行，无默认测试用例失败。​

### 4. 代码规范认知​

- 阅读项目的[开源代码引用规范](https://gitee.com/mindspore/vllm-mindspore/issues/ICIIBG)，遵循开源社区的要求，尊重他人贡献，履行相关的 LICENSE 义务。​

- vLLM-MindSpore插件代码仓库遵循与vLLM开源社区一致的代码规范，可通过[谷歌Python风格指南](https://google.github.io/styleguide/pyguide.html)和[谷歌C++风格指南](https://google.github.io/styleguide/cppguide.html) ，了解项目的代码风格（如缩进、命名规则、注释要求等）。使用vLLM社区代码检查工具：yapf、codespell、ruff、isort和mypy，建议参考[门禁codecheck处理指导书](https://gitee.com/mindspore/vllm-mindspore/issues/ICTIAH)安装项目推荐的代码检查工具。​

## 二、贡献类型及流程

### 1. 提交 Bug 修复​
**流程：**

**1. 搜索 Issue：** 先在项目Issues中搜索相关 Bug，确认是否已被报告。若未报告，新建Issue，选择 “Bug Report” 模板，清晰描述 Bug 现象、复现步骤、预期结果及环境信息（如系统版本、依赖版本）。​

**2. 代码修复：** 参考[Fork-Pull开发模型](#Fork-Pull开发模型), 在本地分支中修复 Bug，确保只修改与 Bug 相关的代码，避免无关改动。​

**3. 测试验证：** 新增或修改对应的测试用例，确保 Bug 已修复且未引入新问题。​

**4. 提交 PR：** 将本地分支推送到远程仓库，提交 Pull Request（PR），PR 标题格式：`[Bugfix]修复Bug简要描述`，正文需关联对应的Issue编号（如Fixes #ABC123），并说明修复思路。​

### 2. 新增功能开发​​
**流程：**

**1. 功能提案：** 若为新增功能，先在Issues中新建 “Feature Request/RFC”，描述功能需求、使用场景及实现思路等内容，等待社区维护者（maintainer）确认可行性。Feature Request 是轻量需求提议，RFC 偏向结构化方案提案。

**2. 代码开发：** 经维护者确认后，参考[Fork-Pull开发模型](#Fork-Pull开发模型)，按照项目代码规范实现功能，确保代码可读性和可维护性，同时编写详细的注释。​

**3. 测试覆盖：** 为新增功能编写完整的测试用例（单元测试、集成测试等），确保功能稳定运行，测试覆盖率不低于80%。（测试用例指导文档撰写中，敬请关注官网更新。当前可参考测试用例目录写法。）

**4. 文档更新：** 特性设计文档需提交至社区Wiki归档（如有），同步更新项目文档（如README.md等），确保文档与功能一致。​

**5. 提交 PR：** 推送分支并提交 PR，PR 标题格式：`[Feat/RFC] 新增功能简要描述`，正文关联 Feature Request/RFC 的Issue编号，说明功能实现细节及测试情况。​

### 3. 贡献新模型
贡献新模型整体适配流程遵循新增功能开发流程，若希望将一个新模型合入vLLM-MindSpore插件代码仓库，需要注意几点：

- **文件格式及位置要遵循规范。** 模型代码文件统一放置于`vllm_mindspore/model_executor`文件夹下，请根据不同模型将代码文件放置于对应的文件夹下。
- **模型基于MindSpore接口实现，支持jit静态图方式执行。** vLLM-MindSpore插件中的模型定义需基于MindSpore接口实现。由于MindSpore静态图模式执行性能有优势，因此模型需支持@jit静态图方式执行。详细可参考[Qwen2.5](https://gitee.com/mindspore/vllm-mindspore/blob/master/vllm_mindspore/model_executor/models/qwen2.py)模型定义实现。
- **将新模型在vLLM-MindSpore插件代码中进行注册。** 模型结构定义实现后，需要将该模型注册到vLLM-MindSpore插件中，注册文件位于`vllm_mindspore/model_executor/models/registry.py`中，请将模型注册到`_NATIVE_MODELS`。
- **编写单元测试。** 新增的模型需同步提交单元测试用例，用例编写请参考[Qwen2.5模型用例](https://gitee.com/mindspore/vllm-mindspore/blob/master/tests/st/python/cases_parallel/vllm_qwen_7b.py)。

### 4. 文档优化 / 翻译​
**流程：**

**1. 确认需求：** 可直接在Issues中认领 “文档优化” 任务，或新建Issue提出文档改进建议（如补充说明、修正错误、新增翻译版本），选择 “Documentation” 模板。​

**2. 文档修改：** 按照项目文档规范进行修改，确保语言准确、逻辑清晰，翻译内容需符合目标语言的表达习惯。​

**3. 提交 PR：** 推送分支并提交 PR，PR 标题格式：`[Docs]文档优化内容`（如`[Docs]补充API参数说明`、`[Docs]新增中文文档翻译`），正文说明修改内容及优化点。​

### 5. 其他贡献
- 代码重构：非功能特性、非缺陷修复的更改，例如代码重构、版本升级或工具更新等可通过在Issues中新建 “Task Tracking” 提交。

- 测试贡献：参与项目测试，在Issues中反馈测试过程中发现的潜在问题，或完善现有测试用例。​

- 问题反馈：若发现未被报告的问题，按Bug报告规范提交Issue，帮助项目定位问题。​

- 社区支持：在Issues评论区解答其他用户的疑问，参与社区技术交流。​

## 三、代码开发、PR 审核与合并规则​

推荐大家使用Fork-Pull开发模型流程进行代码提交,具体步骤如下:

<a id="Fork-Pull开发模型"></a>
### Fork-Pull开发模型

- Fork vLLM-MindSpore插件代码仓库

    在提交代码至项目之前，请确保已fork此项目到您自己的代码仓库。vLLM-MindSpore插件代码仓库和您自己的代码仓库之间可能会并行开发，请注意它们之间的一致性。

- 克隆远程代码仓库

    如果您想将代码下载到本地计算机，最好使用git方法：

    ```shell
    git clone https://gitee.com/{insert_your_forked_repo}/vllm-mindspore.git
    cd vllm-mindspore
    git remote add upstream https://gitee.com/mindspore/vllm-mindspore.git
    ```

- 本地开发代码。

    为避免分支不一致，建议切换到新分支：

    ```shell
    git checkout -b {new_branch} origin/master
    ```

    以master分支为例，如果需要创建版本分支和下游开发分支，请先修复上游的bug，
    再更改代码。

- 将代码推送到远程代码仓库。

    更新代码后，以正式的方式推送更新：

    ```shell
    git add .
    git status # 查看更新状态。
    git commit -m "你的commit标题"
    git commit -s --amend # 添加commit的具体描述。
    git push origin {new_branch}
    ```

- 将请求拉取到vLLM-MindSpore插件代码仓库。

    在最后一步中，您需要在新分支和vLLM-MindSpore插件仓库主分支之间拉取比较请求。提交PR后，需要在评论中通过`/retest`手动触发门禁检查，进行构建测试。经社区维护者（maintainer）检视通过后可合入。拉取请求应该尽快合并到上游master分支中，以降低合并的风险。

### PR 审核与合并规则

1. 审核流程：提交 PR 后，维护者（maintainer）将进行代码审核，可能会提出修改建议，贡献者需根据建议及时修改并推送更新。​

2. 合并条件：

    - PR 需关联对应的Issue（Bug 修复 / 功能开发），且描述清晰、改动合理，完成Self-checklist自检。​

    - 所有门禁测试用例执行通过，无新增错误。​

    - 代码符合项目规范，无明显语法错误或逻辑问题。​

    - 获得代码所属模块所有维护者（maintainer）的 “审核通过” 评审意见（部分模块可能超过1个维护者/PR涉及多个模块）。​
  
    注意:规模较大的关键特性合入前需要在SIG例会进行充分交流沟通，并提供特性设计文档。

3. 冲突处理：若 PR 与主分支存在代码冲突，贡献者需在本地分支执行`git pull --rebase upstream master`解决冲突后，重新推送分支。​

## 四、社区沟通与支持​

1. 沟通渠道：

    - 优先通过项目 Gitee 的Issues和Pull Requests进行任务沟通和问题反馈。​
  
    - 已支持特性相关设计文档可以在社区Wiki中获取（不断完善中）。​
  
    - 新特性推荐通过SIG例会进行交流沟通：
  
      - 欢迎加入[LLM Infercence Serving SIG](https://www.mindspore.cn/community/SIG)，参与开源项目共建和产业合作
      - SIG例会，双周周三或周四下午，16:30 - 17:30 (UTC+8, [查看您的时区](https://dateful.com/convert/gmt8?t=15))

    - 若需实时交流，可加入项目SIG沟通群组（每次SIG理会结束后会提供最新群组二维码，届时可扫码加入）。​

2. 行为准则：遵守开源社区规范，保持友好、尊重的沟通态度，不进行人身攻击、恶意评论，理性讨论技术问题。​

3. 问题咨询：若在贡献过程中遇到环境搭建、流程疑问等问题，可在Issues中新建 “Questions” 类型的Issue，或在社区群组中提问，维护者和其他社区成员会尽力提供帮助。​

## 五、贡献者权益​

1. 所有被合并的有效贡献（代码、文档、测试等），将被记录在项目[贡献者](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/RELEASE.html)中，认可你的贡献。​

2. 长期积极贡献且表现优秀的成员，将有机会成为项目的模块维护者（maintainer），参与项目决策和日常维护。​

3. 部分项目可能会为优秀贡献者提供周边礼品、技术认证等激励（具体以项目实际安排为准）。​

## 六、注意事项​

1. 不提交与项目无关的 PR，避免重复造轮子（先确认功能是否已存在或规划中）。​

2. 不修改项目的核心架构或基础依赖（除非经维护者明确同意并讨论通过）。​

3. 提交的代码需为原创或符合项目的开源协议，不包含侵权、涉密或恶意代码。​

4. 若 PR 长时间未被审核，可在 PR 评论区 @维护者提醒，但请避免频繁催促。​

感谢你的热情参与和支持！让我们一起共建更优秀的vLLM-MindSpore插件开源社区～
