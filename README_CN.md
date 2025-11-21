# MS-InferRT

## 构建说明

### 环境依赖

- GCC >= 9.0
- build
- setuptools
- packaging>=24.2
- wheel>=0.34.2
- pybind11
- nanobind >= 2.9

更多依赖详见`requirements_ascend.txt`，构建生成的whl包将输出到`output`目录。

### 构建方式

#### CPU后端（默认构建）

```bash
bash build.sh
```

#### Ascend后端

```bash
bash build.sh -b ascend
```

#### 开启编译优化能力

```bash
bash build.sh -O
```

#### 并行构建

```bash
bash build.sh -j32
```

设置并行构建任务数（默认为8），可显著提升构建速度。

#### UT构建

```bash
bash build.sh -t
```

## 测试验证

### UT

#### UT执行命令

```bash
bash tests/ut/runtest.sh cpp
```

#### 更多选项

```bash
bash build.sh -h
```

### ST

#### 测试进程说明

- 每个`@arg_mark`标记的`def test_xxx()`函数启动独立进程
- 多个参数组合同一进程，涉及进程内资源共享：图编译缓存、环境变量等
- 用例责任人需手动通过`os.system`等方式在Python脚本中管理多进程

#### ST执行命令

##### Ascend

```bash
bash tests/st/runtest.sh ascend
```

##### CPU

```bash
bash tests/st/runtest.sh cpu
```
