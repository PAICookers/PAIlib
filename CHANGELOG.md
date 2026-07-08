# 变更日志

## v2.0.0

### 破坏性变更

- 用更清晰的 `core_*`、`neuron_*` 模块替代旧的 `reg_*`、`ram_*` 公共模块名
- 包级导出改为显式公共符号，不再依赖模块级通配符重导出
- 参数术语文档重命名为 `docs/Glossary.md`

### 新增

- 新增 PAICORE 2.5 硬件常量、坐标模型、符号-数值转换辅助函数与 AER 路由工具
- 新增 PAICORE 2.5 离线核、在线核的计算核/神经元寄存器模型
- 新增 PAICORE 2.5 帧生成方法
- 新增 PAICORE 2.5 帧流解析工具
- 新增 BF16/FP32 参数载体与浮点载荷打包工具
- 新增面向路由求解器的坐标、AER 路由路径、全局信号方向编码等辅助函数

### 工程化

- 迁移到 `src/` 布局、PEP 621 项目元数据和 `uv_build` 构建后端
- 更新 Ruff 与 pre-commit 集成
- 补充 PAICORE 2.5 模型、路由、全局信号、帧生成和帧解析的测试覆盖

## v1.0.0

- 为 `ReplicaionId` 新增魔法方法
- 为 `CoordOffset` 新增 `from_offset` 方法

## v1.1.0

Yanked release

## v1.1.1

- 添加基础路由类型以支持多芯片

## v1.1.2

- 修复 `bit_split` 截取高位错误的问题

## v1.1.3

- 移除 Python 3.8 支持

## v1.1.4

- 修复 `Coordinate` 加减法进位错误的问题

## v1.1.5

- 生成初始化帧支持芯片列表
- 重构 `CoreMode`

## v1.1.6

- 移除 `hw_types`
- 支持神经元泄露参数为数组形式与导出
- 提供更多的参数模型常量

## v1.2.0

- `Coord` 添加 `core_type` 属性，用于标识坐标所指处理核类型
- 新增 `RoutingPath`，用于表示路由路径

## v1.3.0

- 更新[参数术语文档](docs/Table-of-Terms.md)
- 新增 `FANOUT_IW8` 硬件参数，更正 `ADDR_RAM_MAX` 值为511
- 核模式 `MODE_BANN_OR_SNN_TO_SNN` 重命名为 `MODE_BANN_OR_SNN_TO_VSNN`，并为 `CoreMode` 新增了几个辅助属性
- 核参数权值精度 `WeightPrecisionType` 重命名为 `WeightWidthType`
- 修复 `Frame` 与 `FramePackage` 的深拷贝方法
- 配置帧III型生成方法接口变更：不再接收 `lcn_ex` 与 `weight_width` 参数，改为直接接收 `repeat` 参数
- 重命名部分生成帧方法的参数名

## v1.3.1

- 新增在线核起始坐标 `ONLINE_CORES_BASE_COORD`

## v1.4.0

- 修复错误的测试帧属性
- `get_replication_id` 函数接口变更，现在还会返回多播的基坐标
- 为 `Coord` 新增辅助打印方法

## v1.4.1

- 统一离线核膜电平参数接口，支持离线核测试帧3型输出帧设置膜电平参数 `voltage`

## v1.5.0

- 支持在线核
- 重构帧类、重构参数检查模型

## v1.5.1

- 将部分方法移动到 `ChipFrameGen` 中
- 支持 `uv`
