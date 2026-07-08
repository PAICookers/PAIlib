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

`paicorelib` is the Python support library for the PAICORE series. It provides
foundational capabilities including coordinates, routing, register models,
frame generation, frame parsing, and payload packing.

[中文](README.md)

## Installation

```bash
pip install paicorelib
```

Python 3.10 or newer is required.

## Highlights

- Coordinate, hardware parameter, core, and neuron definitions
- Register parameter models
- Routing utilities
- Global-signal direction encoding utilities
- Frame generation utilities
- PAICORE 2.5 configuration frame stream parsing utilities
- Dense, CSC, BF16, and FP32 weight and payload packing utilities

## Project Layout

```text
src/paicorelib/
├─ __init__.py          # package-level public exports
├─ coordinate.py        # coordinates, offsets, and coordinate utilities
├─ hw_defs.py           # hardware parameters and constants
├─ core_defs*.py        # core register enums and limits
├─ core_model*.py       # core register parameter models
├─ neuron_defs*.py      # neuron register enums and limits
├─ neuron_model*.py     # neuron register parameter models
├─ routing_defs.py      # basic PAICORE multicast routing types
├─ routing_hexa.py      # PAICORE 2.5 NoC / AER routing and global-signal encoding utilities
├─ float_codec.py       # BF16/FP32 value carriers and payload encoding utilities
└─ framelib/
    ├─ frame_defs.py    # frame formats and field definitions
    ├─ frames.py        # frame object definitions
    ├─ frame_gen*.py    # frame generation utilities
    ├─ parser_v2.py     # PAICORE 2.5 configuration frame stream parser
    ├─ types.py         # frame array and payload types
    └─ utils.py         # frame payload helpers
```

## Documentation

- [Parameter glossary](docs/Glossary.md)
- [Changelog](CHANGELOG.md)

## Development

```bash
uv sync --group dev
uv run pytest -q
```
