<h1 align="center">
vLLM-MindSpore Plugin Community Contribution Guide
</h1>

Welcome to the vLLM-MindSpore Plugin community! Whether you're submitting code, fixing bugs, refining documentation, or proposing feature ideas, every contribution you make helps elevate this project. This guide outlines the complete contribution workflow, specification requirements, and support channels to ensure a smooth and efficient experience for all contributors.

## I. Pre-Contribution Preparation
### 1. Contributor License Agreement (CLA)

- The vLLM-MindSpore Plugin is part of the MindSpore community. It's required to sign CLA before your first code submission to MindSpore community. For individual contributor, please refer to [ICLA online document](https://www.mindspore.cn/icla) for the detailed information.

### 2. Project Familiarization
- Thoroughly review the [README.md](https://gitee.com/mindspore/vllm-mindspore/blob/master/README_en.md) in the project root directory to understand the project's positioning, core functionalities, and technology stack.

- Check the [Release Notes](https://www.mindspore.cn/vllm_mindspore/docs/en/master/RELEASE.html) and [Milestones](https://gitee.com/mindspore/vllm-mindspore/milestones) to gain insights into the project's update history and current development priorities.

### 3. Environment Setup

- Follow the installation steps in the [Installation Guide](https://www.mindspore.cn/vllm_mindspore/docs/en/master/getting_started/installation/installation.html) to set up your local development environment.

- Validate your environment by running the project's local test command: execute `pytest tests` in the root directory of the code repository (note: you may need to update the path to your local model weights). Ensure all default test cases pass to confirm the environment is functioning correctly.

### 4. Coding guidelines​

- Review the project's [Open Source Code Citation Specifications](https://gitee.com/mindspore/vllm-mindspore/issues/ICIIBG). Adhere to open source community norms, respect the contributions of others, and fulfill obligations outlined in the project's LICENSE.

- The vLLM-MindSpore Plugin adheres to the same code standards as the vLLM community. You can learn about the project's code style (indentation, naming conventions, comment requirements, etc.) through the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Utilize the vLLM community's code checking tools: yapf, codespell, ruff, isort, and mypy. For installation guidance, refer to the [CI Codecheck Processing Guide](https://gitee.com/mindspore/vllm-mindspore/issues/ICTIAH).

## II. Contribution Types & Workflows

### 1. Bug Fix Submission
**Workflow:**

**1. Search Existing Issues:** First, search the project's Issues to check if the bug has already been reported. If not, create a new Issue using the "Bug Report" template. Clearly describe the bug behavior, reproduction steps, expected results, and environment details (e.g., system version, dependency versions).

**2. Code Fix Implementation:** Refer to the [Fork-Pull development model](#Fork-Pull development model), to fix the bug in a local branch. Focus only on bug-related changes to avoid unnecessary modifications.

**3. Test Validation:** Add or modify corresponding test cases to ensure the bug is fixed and no new issues are introduced.

**4. Submit Pull Request (PR):** Push your local branch to the remote repository and submit a PR. Use the title format: `[Bugfix] Brief description of the bugfix`. In the PR body, link to the associated Issue (e.g., "Fixes #ABC123") and explain your fix approach.

### 2. New Feature Development
**Workflow:**

**1. Feature Proposal:** For new features, first create a "Feature Request/RFC" Issue. Clearly outline the functional requirements, use cases, and implementation ideas. Wait for confirmation of feasibility from community maintainers. Feature Requests are lightweight requirement proposals, whereas RFCs are more inclined toward structured solution proposals.

**2. Code Development:** After confirmation by maintainers, refer to the [Fork-Pull development model](#Fork-Pull development model) to implement the feature. Adhere to project code standards, ensure code readability and maintainability, and include detailed comments.

**3. Test Coverage:** Write comprehensive test cases (unit tests, system tests, etc.) for the new feature to ensure stability. Test coverage must be at least 80%. (The test case guide document is being written; please pay attention to official website updates. Currently, refer to existing test cases in the tests directory.)

**4. Documentation Update:** Submit feature design documents to the community Wiki for archiving (if exists). Synchronously update project documentation (e.g., README.md) to ensure consistency with the new feature.

**5. Submit PR:** Push your branch and submit a PR. Use the title format: `[Feat/RFC] Brief description of the new feature`. In the PR body, link to the Feature Request/RFC Issue and explain key implementation details and test results.

### 3. New Model Contribution
The workflow for contributing new models aligns with the new feature development process. If you want to merge a new model into the vLLM-MindSpore Plugin code repository, pay attention to the following points:

- **File Format & Location Compliance:** All model code files must be placed in the `vllm_mindspore/model_executor` directory. Organize files into subdirectories based on model types.
- **MindSpore Interface & JIT Static Graph Support:** Model definitions in the vLLM-MindSpore Plugin code repository must be implemented based on MindSpore interfaces. Since MindSpore's static graph mode offers better execution performance, models must support execution via the @jit static graph method. For details, refer to the model definition implementation of [Qwen2.5](https://gitee.com/mindspore/vllm-mindspore/blob/master/vllm_mindspore/model_executor/models/qwen2.py).
- **Model Registration:** After implementing the model structure definition, register the model in the vLLM-MindSpore Plugin. The registration file is located at `vllm_mindspore/model_executor/models/registry.py`, please register the model in `_NATIVE_MODELS`.
- **Unit Test Requirements:** Submit unit tests for the new model. Refer to the [Qwen2.5 model test cases](https://gitee.com/mindspore/vllm-mindspore/blob/master/tests/st/python/cases_parallel/vllm_qwen_7b.py) as a template.

### 4. Documentation Refinement / Translation​
**Workflow:**

**1. Confirm Requirements:** Either claim an existing "Documentation Optimization" task in Issues or create a new Issue using the "Documentation" template to propose improvements (e.g., adding explanations, correcting errors, or providing new translations).

**2. Document Modification:** Make modifications in accordance with the project's documentation specifications to ensure accurate language and clear logic. Translated content must conform to the expression habits of the target language.

**3. Submit PR:** Push your branch and submit a PR. Use the title format: `[Docs] Brief description of the change` (e.g., `[Docs] Add API parameter explanations`, `[Docs] Add English translation for Chinese documentation`). In the PR body, explains the modified content and optimization points.

### 5. Other Contributions
- Code Refactoring: Changes that are neither functional features nor defect fixes (such as code refactoring, version upgrades, or tool updates) can be submitted by creating a "Task Tracking" in Issues.

- Testing Contributions: Participate in project testing by reporting potential issues in Issues or enhancing existing test cases.

- Issue Reporting: If you find an unreported problem, submit an Issue in accordance with the "Bug Report" template to help address problems promptly.

- Community Support: Answer other users' questions in Issue comments and participate in community technical discussions.​

## III. Code Development, PR Review & Merging Rules​

We recommend using the Fork-Pull Development Model for code submission. The detailed workflow is as follows:

<a id="Fork-Pull development model"></a>
### Fork-Pull development model

- Fork vLLM-MindSpore Plugin repository

    Before submitting code to vLLM-MindSpore Plugin project, please make sure that this project have been forked to your own repository. It means that there will be parallel development between vLLM-MindSpore Plugin repository and your own repository, so be careful to avoid the inconsistency between them.

- Clone the remote repository

    If you want to download the code to the local machine, `git` is the best way:

    ```shell
    git clone https://gitee.com/{insert_your_forked_repo}/vllm-mindspore.git
    cd vllm-mindspore
    git remote add upstream https://gitee.com/mindspore/vllm-mindspore.git
    ```

- Develop code locally

    To avoid inconsistency between multiple branches, checking out to a new branch is `SUGGESTED`:

    ```shell
    git checkout -b {new_branch_name} origin/master
    ```

    Taking the master branch as an example, vLLM-MindSpore Plugin may create version branches and downstream development branches as needed, please fix bugs upstream first. Then you can change the code arbitrarily.

- Push the code to the remote repository

    After updating the code, you should push the update in the formal way:

    ```shell
    git add .
    git status # Check the update status
    git commit -m "Your commit title"
    git commit -s --amend #Add the concrete description of your commit
    git push origin {new_branch_name}
    ```

- Pull a request to vLLM-MindSpore Plugin repository

    In the last step, your need to pull a compare request between your new branch and vLLM-MindSpore Plugin `master` branch. After submission, manually trigger CI checks with /retest in the comments. PRs should be merged into upstream master promptly to minimize merge risks.
  
### PR Review & Merging Rules

1. Review Process: After submitting a PR, maintainers will conduct code reviews and may provide modification suggestions. Contributors should address comments promptly and push updates to the same branch.

2. Merging Criteria:

    - The PR must be linked to the corresponding Issue (bug fix / feature development), with a clear description, reasonable changes, and completion of the Self-checklist.
  
    - All CI test cases pass without new faults.

    - The code must comply with project specifications, with no syntax errors or logical flaws.

    - Obtain "Pass review" support from all maintainers of the relevant module(s) (some modules may have more than one maintainer / the PR may be involved multiple modules).
  
    Note: For large-scale features, full communication and discussion are required in the SIG meeting before merging, and a feature design document must be provided.

3. Conflict Resolution: If there are code conflicts between the PR and the master branch, contributors need to execute `git pull --rebase upstream master` in the local branch to resolve the conflicts, then push the branch again.

## IV. Community Communication & Support

1. Communication Channels:
    - Primary Channels: Use Gitee Issues and Pull Requests for task coordination and issue reporting.
  
    - Design Documents: Design documents for supported features are available in the community Wiki (continuously updated).
  
    - SIG Meetings: New features are strongly recommended to be discussed in SIG meetings:
  
      - Welcome to join [LLM Infercence Serving SIG](https://www.mindspore.cn/community/SIG) to participate in the co-construction of open-source projects and industrial cooperation
  
      - SIG meetings, every other Wednesday or Thursday afternoon, 16:30 - 17:30 (UTC+8,   [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))

    - Real-Time Communication: Join the LLM Infercence Serving SIG communication group (the latest QR code will be shared at the end of each SIG meeting).​

2. Code of Conduct: Adhere to open source community ethics, maintain a friendly and respectful attitude, refrain from personal attacks or malicious comments, and engage in rational technical discussions.

3. Support & Consultation: If you encounter issues with environment setup, workflow questions, or technical problems, create a "Questions" type Issue or ask in the community group. Maintainers and community members will provide assistance promptly.

## V. Contributor Rights and Interests

1. All valid merged contributions (code, documentation, tests, etc.) will be recorded in the project's [Contributors](https://www.mindspore.cn/vllm_mindspore/docs/en/master/RELEASE.html) to recognize your contribution.

2. Long-term active contributors with outstanding performance will have the opportunity to become module maintainers of the project, participating in project decision-making and daily maintenance.

3. Some projects may provide incentives such as peripheral gifts and technical certifications for outstanding contributors (subject to the project's actual arrangements).​

## VI. Notes

1. Do not submit PRs unrelated to the project. Avoid duplicating existing functionalities — confirm if a feature exists or is planned before development.​

2. Do not modify the project's core architecture or foundational dependencies without explicit approval and discussion with maintainers.

3. Submitted code must be original or comply with the project's open source license. Do not include infringing, confidential, or malicious code.

4. If your PR is pending review for an extended period, gently remind maintainers by @mentioning them in the PR comments. Avoid frequent follow-ups.

Thank you for your enthusiastic participation and support! Let's collaborate to build an even stronger vLLM-MindSpore Plugin community.
