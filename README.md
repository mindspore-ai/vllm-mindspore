# MS-InferRT

## Description

DART project is a collection of programming language, intermediate representation, and runtime, designed for AI (abbreviated as DA). The key task is to provide a light-weight and high-performance runtime for MindSpore in the inference phase. It's under development by now.

## Software Architecture

![dart architecture](./docs/architecture_dart.svg)

## Instructions

### Standard build

```bash
bash build.sh
```

### Build with tests

```bash
bash build.sh -t
```

### Run tests

Run system tests

```bash
bash tests/st/runtest.sh {cpu|ascend}
```

Run unit tests

```bash
bash tests/ut/runtest.sh {cpp|python}
```