# Table of Terms

## 离线核寄存器参数

### 离线核

| 参数名（手册）  |        含义        | 参数模型检验名  | 参数模型导出名(json) |
| :-------------: | :----------------: | :-------------: | :------------------: |
|  weight_width   | crossbar的权重精度 |  weight_width   |     weight_width     |
|       LCN       |    扇入扩展规模    |       lcn       |         lcn          |
|   input_width   |    输入数据位宽    |   input_width   |     input_width      |
|   spike_width   |    输出数据位宽    |   spike_width   |     spike_width      |
|   neuron_num    |   有效树突数量\*   |  num_dendrite   |      neuron_num      |
|    pool_max     |    最大池化使能    | max_pooling_en  |       pool_max       |
| tick_wait_start |     核启动时间     | tick_wait_start |   tick_wait_start    |
|  tick_wait_end  |   核工作持续时间   |  tick_wait_end  |    tick_wait_end     |
|     SNN_EN      |    SNN模式使能     |     snn_en      |        snn_en        |
|   target_LCN    |  输出目标核的LCN   |   target_lcn    |      target_lcn      |
| test_chip_addr  |  测试帧的目标地址  | test_chip_addr  |    test_chip_addr    |

\*手册的文字描述并不准确

### 在线核

|   参数名（手册）   |               含义                |   参数模型检验名   | 参数模型导出名(json) |
| :----------------: | :-------------------------------: | :----------------: | :------------------: |
|     bit_select     |        crossbar的权重精度         |    weight_width    |      bit_select      |
|    group_select    | 控制多个神经元的组合（扇入倍增）  |        lcn         |     group_select     |
| lateral_inhi_value |         侧抑制的数值大小          | lateral_inhi_value |  lateral_inhi_value  |
| weight_decay_value |        权值衰退的数值大小         | weight_decay_value |  weight_decay_value  |
|    upper_weight    |      在线学习的权值更新上界       |    upper_weight    |     upper_weight     |
|    lower_weight    |      在线学习的权值更新下界       |    lower_weight    |     lower_weight     |
|    neuron_start    |       有效神经元的起始序号        |    neuron_start    |     neuron_start     |
|     neuron_end     |       有效神经元的结束序号        |     neuron_end     |      neuron_end      |
|  inhi_core_x_star  | 侧抑制需要用X\*地址复制的影响范围 |   inhi_core_x_ex   |   inhi_core_x_star   |
|  inhi_core_y_star  | 侧抑制需要用Y\*地址复制的影响范围 |   inhi_core_y_ex   |   inhi_core_y_star   |
|  core_start_time   |            核启动时间             |  tick_wait_start   |   core_start_time    |
|   core_hold_time   |          核工作持续时间           |   tick_wait_end    |    core_hold_time    |
|   LUT_random_en    |       LUT查找表随机更新使能       |   lut_random_en    |    lut_random_en     |
|  decay_random_en   |       权值衰退随机更新使能        |  decay_random_en   |   decay_random_en    |
|   leakage_order    |         阈值比较前/后泄漏         |     leak_order     |    leakage_order     |
|   online_mode_en   |    执行在线学习或离线推断模式     |   online_mode_en   |    online_mode_en    |
|    test_address    |         测试帧的目标地址          |   test_chip_addr   |     test_address     |
|    random_seed     |          非零随机数种子           |    random_seed     |     random_seed      |

## 神经元寄存器参数

### 离线核神经元

|   参数名（手册）    |          含义           |    参数模型检验名     | 参数模型导出名(json) |
| :-----------------: | :---------------------: | :-------------------: | :------------------: |
|    tick_relative    |      相对时间信息       |     tick_relative     |    tick_relative     |
|      addr_axon      |        目标轴突         |       addr_axon       |      addr_axon       |
|     addr_core_x     |       目标核X地址       |      addr_core_x      |     addr_core_x      |
|     addr_core_y     |       目标核Y地址       |      addr_core_y      |     addr_core_y      |
|   addr_core_x_ex    |     目标X复制位标识     |    addr_core_x_ex     |    addr_core_x_ex    |
|   addr_core_y_ex    |     目标Y复制位标识     |    addr_core_y_ex     |    addr_core_y_ex    |
|     addr_chip_x     |     目标芯片的X地址     |      addr_chip_x      |     addr_chip_x      |
|     addr_chip_y     |     目标芯片的Y地址     |      addr_chip_y      |     addr_chip_y      |
|     reset_mode      |      复位模式选择       |      reset_mode       |      reset_mode      |
|       reset_v       |      膜电平复位值       |        reset_v        |       reset_v        |
|      leak_post      |    阈值比较前/后泄漏    |    leak_comparison    |      leak_post       |
| threshold_mask_ctrl |        阈值掩码         |    thres_mask_bits    | threshold_mask_ctrl  |
| threshold_neg_mode  |     负阈值模式选择      |    neg_thres_mode     |  threshold_neg_mode  |
|    threshold_neg    |         负阈值          |     neg_threshold     |    threshold_neg     |
|    threshold_pos    |         正阈值          |     pos_threshold     |    threshold_pos     |
| leak_reversal_flag  |    反向泄漏模式选择     |    leak_direction     |  leak_reversal_flag  |
|   leak_det_stoch    |    泄漏随机模式选择     | leak_integration_mode |    leak_det_stoch    |
|       leak_v        |        泄漏电平         |        leak_v         |        leak_v        |
|  weight_det_stoch   |    权重随机模式选择     | syn_integration_mode  |   weight_det_stoch   |
|    bit_truncate     |     膜电平截取位置      |       bit_trunc       |     bit_truncate     |
|       vjt_pre       | 膜电平（只读，复位值0） |        voltage        |       voltage        |

### 在线核神经元

在 1-bit 权值精度下，由一个神经元地址存储参数，共128bits；2-/4-/8-bit 权值精度下，由两个神经元地址存储参数，共256bits。

|    参数名（手册）     |                     含义                     |  参数模型检验名  | 参数模型导出名(json)  |
| :-------------------: | :------------------------------------------: | :--------------: | :-------------------: |
|      leakage_reg      |                   泄漏电位                   |      leak_v      |      leakage_reg      |
|     threshold_reg     |                  神经元阈值                  |    threshold     |     threshold_reg     |
|  floor_threshold_reg  |                神经元地板阈值                |   floor_thres    |  floor_threshold_reg  |
|  reset_potential_reg  |                 膜电平复位值                 |     reset_v      |  reset_potential_reg  |
| initial_potential_reg | 初始膜电平，每次开始帧到达时膜电平复位为该值 |      init_v      | initial_potential_reg |
|     potential_reg     |                  当前膜电平                  |     voltage      |     potential_reg     |
|       time_slot       |                 相对时间信息                 |  tick_relative   |     tick_relative     |
|      addr_chip_x      |               目标芯片的X地址                |   addr_chip_x    |      addr_chip_x      |
|      addr_chip_y      |               目标芯片的Y地址                |   addr_chip_y    |      addr_chip_y      |
|      addr_core_x      |                 目标核X地址                  |   addr_core_x    |      addr_core_x      |
|      addr_core_y      |                 目标核Y地址                  |   addr_core_y    |      addr_core_y      |
|   addr_core_x_star    |               目标X复制位标识                |  addr_core_x_ex  |    addr_core_x_ex     |
|   addr_core_y_star    |               目标Y复制位标识                |  addr_core_y_ex  |    addr_core_y_ex     |
|       addr_axon       |                   目标轴突                   |    addr_axon     |       addr_axon       |
|   plasticity_start    |              突触可塑性开始位置              | plasticity_start |   plasticity_start    |
|    plasticity_end     |              突触可塑性结束位置              |  plasticity_end  |    plasticity_end     |
