# vllm 社区 codecheck 检查工具链使用说明

## 使用步骤
- 1. 确保修改已经```git commit```，并合并成一个commit id.
- 2. 运行命令：```bash vllm_codecheck.sh```

## 执行说明
- 1、根据 ``requiremnts-lint.txt``安装工具链，请确保网络畅通。
- 2、依次运行`yaph`, `codespell`, `ruff`, `isort`, `mypy` 工具。

## 工具说明
- `yapf`: 自动formatting工具。
- `codespell`: 拼写检查工具。
- `ruff`: 代码format检查工具。
- `isort`: 自动修复import工具。
- `mypy`: 静态类型检查工具。

## 修复建议：
- `codespell`如需屏蔽拼写错误，修改`pyproject.toml`中的

```commandline
[tool.codespell]
ignore-words-list = "dout, te, indicies, subtile, ElementE, CANN"
```

- `ruff` 如需屏蔽检查，在代码行后增加注释

```commandline
# noqa: {error_code}
```
