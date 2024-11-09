# Table of Terms

## 离线核参数

### 核寄存器参数

| 参数名（手册）  |        含义        |   参数模型检验名   | 参数模型导出名  |
| :-------------: | :----------------: | :----------------: | :-------------: |
|  weight_width   | crossbar的权重精度 |    weight_width    |  weight_width   |
|       LCN       |    扇入扩展规模    |   lcn_extension    |       LCN       |
|   input_width   |    输入数据位宽    | input_width_format |   input_width   |
|   spike_width   |    输出数据位宽    | spike_width_format |   spike_width   |
|   neuron_num    |   有效树突数量\*   |    num_dendrite    |  num_dendrite   |
|    pool_max     |    最大池化使能    |   max_pooling_en   |    pool_max     |
| tick_wait_start |     核启动时间     |  tick_wait_start   | tick_wait_start |
|  tick_wait_end  |   核工作持续时间   |   tick_wait_end    |  tick_wait_end  |
|     SNN_EN      |    SNN模式使能     |    snn_mode_en     |     snn_en      |
|   target_LCN    |  输出目标核的LCN   |     target_lcn     |   target_LCN    |
| test_chip_addr  |  测试帧的目标地址  |   test_chip_addr   | test_chip_addr  |

\*手册的文字描述并不准确

### 神经元寄存器参数

|   参数名（手册）    |       含义        |      参数模型检验名       |   参数模型导出名    |
| :-----------------: | :---------------: | :-----------------------: | :-----------------: |
|    tick_relative    |   相对时间信息    |       tick_relative       |    tick_relative    |
|      addr_axon      |     目标轴突      |         addr_axon         |      addr_axon      |
|     addr_core_x     |    目标核x地址    |        addr_core_x        |     addr_core_x     |
|     addr_core_y     |    目标核y地址    |        addr_core_y        |     addr_core_y     |
|   addr_core_x_ex    |  目标x复制位标识  |      addr_core_x_ex       |   addr_core_x_ex    |
|   addr_core_y_ex    |  目标y复制位标识  |      addr_core_y_ex       |   addr_core_y_ex    |
|     addr_chip_x     |  目标芯片的x地址  |        addr_chip_x        |     addr_chip_x     |
|     addr_chip_y     |  目标芯片的y地址  |        addr_chip_y        |     addr_chip_y     |
|     reset_mode      |   复位模式选择    |        reset_mode         |     reset_mode      |
|       reset_v       |   膜电平复位值    |          reset_v          |       reset_v       |
|      leak_post      | 阈值比较前/后泄露 |      leak_comparison      |      leak_post      |
| threshold_mask_ctrl |     阈值掩码      |    threshold_mask_bits    | threshold_mask_ctrl |
| threshold_neg_mode  |  负阈值模式选择   |      neg_thres_mode       | threshold_neg_mode  |
|    threshold_neg    |      负阈值       |       neg_threshold       |    threshold_neg    |
|    threshold_pos    |      正阈值       |       pos_threshold       |    threshold_pos    |
| leak_reversal_flag  | 反向泄露模式选择  |      leak_direction       | leak_reversal_flag  |
|   leak_det_stoch    | 泄露随机模式选择  |   leak_integration_mode   |   leak_det_stoch    |
|       leak_v        |     泄露电平      |          leak_v           |       leak_v        |
|  weight_det_stoch   | 权重随机模式选择  | synaptic_integration_mode |  weight_det_stoch   |
|    bit_truncate     |  膜电平截取位置   |      bit_truncation       |    bit_truncate     |
|       vjt_pre       | 膜电平（只读，0） |         vjt_init          |      vjt_init       |
