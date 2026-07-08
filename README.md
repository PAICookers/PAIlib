<div align="center">

# Library of PAICORE

</div>

<p align="center">
    <a href="https://github.com/PAICookers/PAIlib/blob/main/pyproject.toml">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/paicorelib">
    </a>
    <a href="https://pypi.org/project/paicorelib/">
        <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/paicorelib?color=pink">
    </a>
    <a href="https://www.codefactor.io/repository/github/PAICookers/PAIlib">
        <img alt="CodeFactor Grade" src="https://img.shields.io/codefactor/grade/github/PAICookers/PAIlib?color=orange">
    </a>
    <a href="https://results.pre-commit.ci/latest/github/PAICookers/PAIlib/main">
        <img alt="pre-commit.ci status" src="https://results.pre-commit.ci/badge/github/PAICookers/PAIlib/main.svg">
    </a>
    <a href="https://codecov.io/gh/PAICookers/PAIlib" >
        <img src="https://codecov.io/gh/PAICookers/PAIlib/graph/badge.svg?token=978U1BIZRE"/>
    </a>
    <a href="https://github.com/PAICookers/PAIlib/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/PAICookers/PAIlib">
    </a>
</p>

`paicorelib` 是 PAICORE 系列的 Python 支持库，提供坐标、路由、寄存器模型、帧生成、帧解析和载荷打包等基础能力。

[English](README_EN.md)

## 安装

```bash
pip install paicorelib
```

需要 Python 3.10 或更新版本。

## 主要能力

- 坐标、硬件参数、计算核和神经元定义
- 寄存器参数模型
- 路由工具
- 全局信号方向编码工具
- 帧生成工具
- PAICORE 2.5 配置帧流解析工具
- Dense、CSC、BF16、FP32 等权重与载荷打包工具

## 文件组织

```text
src/paicorelib/
├─ __init__.py          # 包级公共导出入口
├─ coordinate.py        # 坐标、偏移和坐标工具
├─ hw_defs.py           # 硬件参数和常量定义
├─ core_defs*.py        # 计算核寄存器枚举和限制
├─ core_model*.py       # 计算核寄存器参数模型
├─ neuron_defs*.py      # 神经元寄存器枚举和限制
├─ neuron_model*.py     # 神经元寄存器参数模型
├─ routing_defs.py      # PAICORE 多播路由基础类型
├─ routing_hexa.py      # PAICORE 2.5 NoC / AER 路由和全局信号编码工具
├─ float_codec.py       # BF16/FP32 数值载体和载荷编码工具
└─ framelib/
    ├─ frame_defs.py    # 帧格式和字段定义
    ├─ frames.py        # 帧对象定义
    ├─ frame_gen*.py    # 帧生成工具
    ├─ parser_v2.py     # PAICORE 2.5 配置帧流解析
    ├─ types.py         # 帧数组和载荷类型
    └─ utils.py         # 帧载荷辅助工具
```

## 文档

- [参数术语文档](docs/Glossary.md)
- [变更日志](CHANGELOG.md)

## 开发

```bash
uv sync --group dev
uv run pytest -q
```
