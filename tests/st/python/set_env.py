import os
import sys
from typing import Dict, Optional

mindformers_path = "/home/jenkins/mindspore/testcases/testcases/tests/mindformers"

if mindformers_path not in sys.path:
    sys.path.insert(0, mindformers_path)

current_pythonpath = os.environ.get("PYTHONPATH", "")
if current_pythonpath:
    os.environ["PYTHONPATH"] = f"{mindformers_path}:{current_pythonpath}"
else:
    os.environ["PYTHONPATH"] = mindformers_path


class EnvVarManager:
    def __init__(self):
        self._original_env: Dict[str, Optional[str]] = {}
        self._managed_vars: Dict[str, str] = {}

    def set_env_var(self, var_name: str, value: str) -> None:
        """设置环境变量并记录原始值（如果存在）"""
        if var_name not in self._original_env:
            # 保存原始值，即使它不存在（保存为None）
            self._original_env[var_name] = os.environ.get(var_name)

        os.environ[var_name] = value
        self._managed_vars[var_name] = value

    def unset_env_var(self, var_name: str) -> None:
        """取消设置之前设置的环境变量，恢复原始值"""
        if var_name not in self._original_env:
            raise ValueError(f"Variable {var_name} was not set by this manager")

        original_value = self._original_env[var_name]
        if original_value is not None:
            os.environ[var_name] = original_value
        else:
            if var_name in os.environ:
                del os.environ[var_name]

        del self._original_env[var_name]
        del self._managed_vars[var_name]

    def unset_all(self) -> None:
        """取消设置所有由该管理器设置的环境变量"""
        for var_name in list(self._managed_vars.keys()):
            self.unset_env_var(var_name)

    def get_managed_vars(self) -> Dict[str, str]:
        """获取当前由该管理器管理的所有环境变量       """
        return self._managed_vars.copy()

    def setup_ai_environment(self, env_vars: Dict[str, str]) -> None:
        """设置AI相关的环境变量，使用传入的参数"""
        for var_name, value in env_vars.items():
            self.set_env_var(var_name, value)
