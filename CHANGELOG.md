## v1.0.0rc1

- 修复 `reg_model` 的一处判断逻辑错误

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
