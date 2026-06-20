# Table of Terms

\*手册中部分字段的描述并不准确，代码中有修改。

## v2

### 计算核寄存器参数

#### 离线核

| 参数名（手册） |        含义        | 参数模型检验名 | 参数模型导出名(json) |
| :-------------: | :----------------: | :-------------: | :------------------: |
|  weight_width  | crossbar的权重精度 |  weight_width  |     weight_width     |
|       LCN       |    扇入扩展规模    |       lcn       |         lcn         |
|   input_width   |    输入数据位宽    |   input_width   |     input_width     |
|   spike_width   |    输出数据位宽    |   spike_width   |     spike_width     |
|   neuron_num   |   有效树突数量\*   |  num_dendrite  |      neuron_num      |
|    pool_max    |    最大池化使能    | max_pooling_en |       pool_max       |
| tick_wait_start |     核启动时间     | tick_wait_start |   tick_wait_start   |
|  tick_wait_end  |   核工作持续时间   |  tick_wait_end  |    tick_wait_end    |
|     SNN_EN     |    SNN模式使能    |     snn_en     |        snn_en        |
|   target_LCN   |  输出目标核的LCN  |   target_lcn   |      target_lcn      |
| test_chip_addr |  测试帧的目标地址  | test_chip_addr |    test_chip_addr    |

#### 在线核

|   参数名（手册）   |               含义               |   参数模型检验名   | 参数模型导出名(json) |
| :----------------: | :-------------------------------: | :----------------: | :------------------: |
|     bit_select     |        crossbar的权重精度        |    weight_width    |      bit_select      |
|    group_select    | 控制多个神经元的组合（扇入倍增） |        lcn        |     group_select     |
| lateral_inhi_value |         侧抑制的数值大小         | lateral_inhi_value |  lateral_inhi_value  |
| weight_decay_value |        权值衰退的数值大小        | weight_decay_value |  weight_decay_value  |
|    upper_weight    |      在线学习的权值更新上界      |    upper_weight    |     upper_weight     |
|    lower_weight    |      在线学习的权值更新下界      |    lower_weight    |     lower_weight     |
|    neuron_start    |       有效神经元的起始序号       |    neuron_start    |     neuron_start     |
|     neuron_end     |       有效神经元的结束序号       |     neuron_end     |      neuron_end      |
|  inhi_core_x_star  | 侧抑制需要用X\*地址复制的影响范围 |   inhi_core_x_ex   |   inhi_core_x_star   |
|  inhi_core_y_star  | 侧抑制需要用Y\*地址复制的影响范围 |   inhi_core_y_ex   |   inhi_core_y_star   |
|  core_start_time  |            核启动时间            |  tick_wait_start  |   core_start_time   |
|   core_hold_time   |          核工作持续时间          |   tick_wait_end   |    core_hold_time    |
|   LUT_random_en   |       LUT查找表随机更新使能       |   lut_random_en   |    lut_random_en    |
|  decay_random_en  |       权值衰退随机更新使能       |  decay_random_en  |   decay_random_en   |
|   leakage_order   |         阈值比较前/后泄漏         |     leak_order     |    leakage_order    |
|   online_mode_en   |    执行在线学习或离线推断模式    |   online_mode_en   |    online_mode_en    |
|    test_address    |         测试帧的目标地址         |   test_chip_addr   |     test_address     |
|    random_seed    |          非零随机数种子          |    random_seed    |     random_seed     |

### 神经元寄存器参数

#### 离线核神经元

|   参数名（手册）   |          含义          |    参数模型检验名    | 参数模型导出名(json) |
| :-----------------: | :---------------------: | :-------------------: | :------------------: |
|    tick_relative    |      相对时间信息      |     tick_relative     |    tick_relative    |
|      addr_axon      |        目标轴突        |       addr_axon       |      addr_axon      |
|     addr_core_x     |       目标核X地址       |      addr_core_x      |     addr_core_x     |
|     addr_core_y     |       目标核Y地址       |      addr_core_y      |     addr_core_y     |
|   addr_core_x_ex   |     目标X复制位标识     |    addr_core_x_ex    |    addr_core_x_ex    |
|   addr_core_y_ex   |     目标Y复制位标识     |    addr_core_y_ex    |    addr_core_y_ex    |
|     addr_chip_x     |     目标芯片的X地址     |      addr_chip_x      |     addr_chip_x     |
|     addr_chip_y     |     目标芯片的Y地址     |      addr_chip_y      |     addr_chip_y     |
|     reset_mode     |      复位模式选择      |      reset_mode      |      reset_mode      |
|       reset_v       |      膜电平复位值      |        reset_v        |       reset_v       |
|      leak_post      |    阈值比较前/后泄漏    |    leak_comparison    |      leak_post      |
| threshold_mask_ctrl |        阈值掩码        |    thres_mask_bits    | threshold_mask_ctrl |
| threshold_neg_mode |     负阈值模式选择     |    neg_thres_mode    |  threshold_neg_mode  |
|    threshold_neg    |         负阈值         |     neg_threshold     |    threshold_neg    |
|    threshold_pos    |         正阈值         |     pos_threshold     |    threshold_pos    |
| leak_reversal_flag |    反向泄漏模式选择    |    leak_direction    |  leak_reversal_flag  |
|   leak_det_stoch   |    泄漏随机模式选择    | leak_integration_mode |    leak_det_stoch    |
|       leak_v       |        泄漏电平        |        leak_v        |        leak_v        |
|  weight_det_stoch  |    权重随机模式选择    | syn_integration_mode |   weight_det_stoch   |
|    bit_truncate    |     膜电平截取位置     |       bit_trunc       |     bit_truncate     |
|       vjt_pre       | 膜电平（只读，复位值0） |        voltage        |       voltage       |

#### 在线核神经元

在 1-bit 权值精度下，由一个神经元地址存储参数，共128 bits；2-/4-/8-bit 权值精度下，由两个神经元地址存储参数，共256 bits。

|    参数名（手册）    |        含义        |  参数模型检验名  | 参数模型导出名(json) |
| :-------------------: | :----------------: | :--------------: | :-------------------: |
|      leakage_reg      |      泄漏电位      |      leak_v      |      leakage_reg      |
|     threshold_reg     |     神经元阈值     |  pos_threshold  |     threshold_reg     |
|  floor_threshold_reg  |   神经元地板阈值   |  neg_threshold  |  floor_threshold_reg  |
|  reset_potential_reg  |    膜电平复位值    |     reset_v     |  reset_potential_reg  |
| initial_potential_reg |     初始膜电平     |      init_v      | initial_potential_reg |
|     potential_reg     |     当前膜电平     |     voltage     |     potential_reg     |
|       time_slot       |    相对时间信息    |  tick_relative  |     tick_relative     |
|      addr_chip_x      |  目标芯片的X地址  |   addr_chip_x   |      addr_chip_x      |
|      addr_chip_y      |  目标芯片的Y地址  |   addr_chip_y   |      addr_chip_y      |
|      addr_core_x      |    目标核X地址    |   addr_core_x   |      addr_core_x      |
|      addr_core_y      |    目标核Y地址    |   addr_core_y   |      addr_core_y      |
|   addr_core_x_star   |  目标X复制位标识  |  addr_core_x_ex  |    addr_core_x_ex    |
|   addr_core_y_star   |  目标Y复制位标识  |  addr_core_y_ex  |    addr_core_y_ex    |
|       addr_axon       |      目标轴突      |    addr_axon    |       addr_axon       |
|   plasticity_start   | 突触可塑性开始位置 | plasticity_start |   plasticity_start   |
|    plasticity_end    | 突触可塑性结束位置 |  plasticity_end  |    plasticity_end    |

## v2.5

### 计算核寄存器参数

#### 离线核

| 参数名（手册） |                                    含义                                    | 参数模型检验名 | 参数模型导出名(json) |
| :------------: | :-------------------------------------------------------------------------: | :------------: | :------------------: |
|    SNN_ANN    |                                 核模式选择                                 |    snn_ann    |       snn_ann       |
|  max_pooling  |                                池化模式选择                                |  max_pooling  |     max_pooling     |
| add_potential |                                累加模式选择                                | add_potential |    add_potential    |
|  zero_output  |                                是否输出零值                                |  zero_output  |     zero_output     |
|   input_sign   |                              输入数据符号使能                              |   input_sign   |      input_sign      |
|  input_width  |                              输入数据位宽选择                              |  input_width  |     input_width     |
|  output_sign  |                              输出数据符号使能                              |  output_sign  |     output_sign     |
|  output_width  |                              输出数据位宽选择                              |  output_width  |     output_width     |
|  weight_sign  |                              权重数据符号使能                              |  weight_sign  |     weight_sign     |
|  weight_width  |                              权重数据位宽选择                              |  weight_width  |     weight_width     |
|      LCN      |                                扇入扩展规模                                |      lcn      |         lcn         |
|   target_LCN   |                               输出目标核的LCN                               |   target_lcn   |      target_lcn      |
|   axon_skew   |                       AER格式输入工作帧的轴突地址偏移                       |   axon_skew   |      axon_skew      |
| neuron_number |          有效神经元地址数量，1全神经元折合2个半神经元（上限4096）          | neuron_number |    neuron_number    |
|  test_core_xy  |                      测试帧/控制帧发送的核的相对xy地址                      |  test_core_xy  |     test_core_xy     |
|  test_core_x  |                      测试帧/控制帧发送的核的相对x地址                      |  test_core_x  |     test_core_x     |
|  test_core_y  |                      测试帧/控制帧发送的核的相对y地址                      |  test_core_y  |     test_core_y     |
|  global_send  |               全局信号的发送方向（local/xy+/xy-/x+/x-/y+/y-）               |  global_send  |     global_send     |
| csc_accelerate |                           csc压缩计算加速模式使能                           | csc_accelerate |    csc_accelerate    |
| global_receive |                      全局信号的接收方向（方向同前述）                      | global_receive |    global_receive    |
| thread_number |                               当前核线程编号                               | thread_number |    thread_number    |
|   busy_cycle   |                              busy信号掩码阈值                              |   busy_cycle   |      busy_cycle      |
|  delay_cycle  |                           控制信号生效的延时时间                           |  delay_cycle  |     delay_cycle     |
|  width_cycle  |           控制全局信号 `sync_all`、`initial_all` 的多周期宽度           |  width_cycle  |     width_cycle     |
|   tick_start   |         当前核在 `tick_start` 次 `sync_all` 时启动，0则永不启动         |   tick_start   |      tick_start      |
| tick_duration |        当前核持续工作 `tick_duration` 次 `sync_all`，0则持续工作        | tick_duration |    tick_duration    |
|  tick_initial  | 当前核工作 `tick_initial` 次 `sync_all` 后自动执行初始化，0则永不初始化 |  tick_initial  |     tick_initial     |

#### 在线核

| 参数名（手册） |                   含义                   | 参数模型检验名 | 参数模型导出名(json) |
| :------------: | :---------------------------------------: | :------------: | :------------------: |
|    SNN_ANN    |             SNN和ANN模式选择             |    snn_ann    |       snn_ann       |
|  max_pooling  |                 同离线核                 |  max_pooling  |     max_pooling     |
| add_potential |                 同离线核                 | add_potential |    add_potential    |
|  zero_output  |                 同离线核                 |  zero_output  |     zero_output     |
|   work_mode   |                核工作模式                |   work_mode   |      work_mode      |
|   input_core   |               输入源核类型               |   input_core   |      input_core      |
|  input_width  |             输入数据位宽类型             |  input_width  |     input_width     |
|  output_core  |              输出目标核类型              |  output_core  |     output_core     |
|  output_width  | 宽前向/梯度核输出数据类型；更新核更新类型 |  output_width  |     output_width     |
|     LCN_AT     |               激活值LCN配置               |     lcn_at     |        lcn_at        |
|     LCN_MP     |               膜电平LCN配置               |     lcn_mp     |        lcn_mp        |
|     LCN_LG     |                梯度LCN配置                |     lcn_lg     |        lcn_lg        |
| target_LCN_AT |           输出目标地址的LCN_AT           | target_lcn_at |    target_lcn_at    |
| target_LCN_MP |           输出目标地址的LCN_MP           | target_lcn_mp |    target_lcn_mp    |
| target_LCN_LG |           输出目标地址的LCN_LG           | target_lcn_lg |    target_lcn_lg    |
|   axon_skew   |                 同离线核                 |   axon_skew   |      axon_skew      |
| neuron_number |                 同离线核                 | neuron_number |    neuron_number    |
| update_number |      需要更新的神经元及权重地址数量      | update_number |    update_number    |
| csc_accelerate |                 同离线核                 | csc_accelerate |    csc_accelerate    |
|    scale_in    |            输入放缩系数(bf16)            |    scale_in    |       scale_in       |
|    bias_in    |            输入偏置系数(bf16)            |    bias_in    |       bias_in       |
|   scale_out   |            输出放缩系数(bf16)            |   scale_out   |      scale_out      |
|    bias_out    |            输出偏置系数(bf16)            |    bias_out    |       bias_out       |
| learning_rate |               学习率(bf16)               | learning_rate |    learning_rate    |
| update_core_xy |       更新权重发送的核的相对xy地址       | update_core_xy |    update_core_xy    |
| update_core_x |        更新权重发送的核的相对x地址        | update_core_x |    update_core_x    |
| update_core_y |        更新权重发送的核的相对y地址        | update_core_y |    update_core_y    |
|  test_core_xy  |                 同离线核                 |  test_core_xy  |     test_core_xy     |
|  test_core_x  |                 同离线核                 |  test_core_x  |     test_core_x     |
|  test_core_y  |                 同离线核                 |  test_core_y  |     test_core_y     |
|  global_send  |                 同离线核                 |  global_send  |     global_send     |
| global_receive |                 同离线核                 | global_receive |    global_receive    |
| thread_number |                 同离线核                 | thread_number |    thread_number    |
|   busy_cycle   |                 同离线核                 |   busy_cycle   |      busy_cycle      |
|  delay_cycle  |                 同离线核                 |  delay_cycle  |     delay_cycle     |
|  width_cycle  |                 同离线核                 |  width_cycle  |     width_cycle     |
|   tick_start   |                 同离线核                 |   tick_start   |      tick_start      |
| tick_duration |                 同离线核                 | tick_duration |    tick_duration    |
|  tick_initial  |                 同离线核                 |  tick_initial  |     tick_initial     |

### 神经元寄存器参数

#### 离线核神经元

**全神经元**与**半神经元**共享参数：

|    参数名（手册）    |            含义            |    参数模型检验名    | 参数模型导出名(json) |
| :------------------: | :------------------------: | :------------------: | :------------------: |
|    tick_relative    |        相对时间信息        |    tick_relative    |    tick_relative    |
|      addr_axon      |        目标轴突地址        |      addr_axon      |      addr_axon      |
|     addr_core_xy     |     目标核的相对xy地址     |     addr_core_xy     |     addr_core_xy     |
|     addr_core_x     |     目标核的相对x地址     |     addr_core_x     |     addr_core_x     |
|     addr_core_y     |     目标核的相对y地址     |     addr_core_y     |     addr_core_y     |
|     addr_copy_xy     |     目标核的xy广播地址     |     addr_copy_xy     |     addr_copy_xy     |
|     addr_copy_x     |     目标核的x广播地址     |     addr_copy_x     |     addr_copy_x     |
|     addr_copy_y     |     目标核的y广播地址     |     addr_copy_y     |     addr_copy_y     |
|     weight_skew     | 神经元对应权重的纵向偏移量 |     weight_skew     |     weight_skew     |
| weight_address_start | 神经元对应权重SRAM开始地址 | weight_address_start | weight_address_start |
|  weight_address_end  | 神经元对应权重SRAM结束地址 |  weight_address_end  |  weight_address_end  |
|     output_type     |          输出类型          |     output_type     |     output_type     |
|      fold_type      |          折叠类型          |      fold_type      |      fold_type      |
|     neuron_type     |         神经元类型         |     neuron_type     |     neuron_type     |
|         vjt         |           膜电平           |         vjt         |         vjt         |

**全神经元**参数：

|   参数名（手册）   |           含义           |   参数模型检验名   | 参数模型导出名(json) |
| :-----------------: | :----------------------: | :-----------------: | :------------------: |
|     reset_mode     |         复位模式         |     reset_mode     |      reset_mode      |
|       reset_v       |       膜电平复位值       |       reset_v       |       reset_v       |
| threshold_neg_mode |      负阈值模式选择      | threshold_neg_mode |  threshold_neg_mode  |
| threshold_pos_mode |      正阈值模式选择      | threshold_pos_mode |  threshold_pos_mode  |
|    threshold_neg    |          负阈值          |    threshold_neg    |    threshold_neg    |
|    threshold_pos    |          正阈值          |    threshold_pos    |    threshold_pos    |
| lateral_inhibition |      侧抑制模式选择      | lateral_inhibition |  lateral_inhibition  |
| leak_multi_sequence |       乘法泄漏顺序       | leak_multi_sequence | leak_multi_sequence |
|  leak_multi_input  |   输入是否参与乘法泄漏   |  leak_multi_input  |   leak_multi_input   |
|   leak_multi_mode   |     乘法泄漏模式选择     |   leak_multi_mode   |   leak_multi_mode   |
|    leak_add_mode    |     加法泄漏模式选择     |    leak_add_mode    |    leak_add_mode    |
|      leak_tau      | 乘法泄漏膜电平左右移位数 |      leak_tau      |       leak_tau       |
|       leak_v       |       加法泄漏电平       |       leak_v       |        leak_v        |
|   weight_compress   |         权重类型         |   weight_compress   |   weight_compress   |
|     vjt_initial     |        初始膜电平        |     vjt_initial     |     vjt_initial     |

**折叠神经元**参数：

| 参数名（手册） |          含义          | 参数模型检验名 | 参数模型导出名(json) |
| :------------: | :--------------------: | :------------: | :------------------: |
| fold_range_xy |     XY维度折叠宽度     | fold_range_xy |    fold_range_xy    |
|  fold_range_x  |     X维度折叠宽度     |  fold_range_x  |     fold_range_x     |
|  fold_range_y  |     Y维度折叠宽度     |  fold_range_y  |     fold_range_y     |
|  fold_skew_xy  |   XY维度权重偏移长度   |  fold_skew_xy  |     fold_skew_xy     |
|  fold_skew_x  |   X维度权重偏移长度   |  fold_skew_x  |     fold_skew_x     |
|  fold_skew_y  |   Y维度权重偏移长度   |  fold_skew_y  |     fold_skew_y     |
|  fold_axon_xy  | XY维度轴突地址偏移步长 |  fold_axon_xy  |     fold_axon_xy     |
|  fold_axon_x  | X维度轴突地址偏移步长 |  fold_axon_x  |     fold_axon_x     |
|  fold_axon_y  | Y维度轴突地址偏移步长 |  fold_axon_y  |     fold_axon_y     |
|  fold_number  |     折叠神经元总数     |  fold_number  |     fold_number     |
|   fold_vjt_3   |  折叠神经元3的膜电平  |   fold_vjt_3   |      fold_vjt_3      |
|   fold_vjt_2   |  折叠神经元2的膜电平  |   fold_vjt_2   |      fold_vjt_2      |
|   fold_vjt_1   |  折叠神经元1的膜电平  |   fold_vjt_1   |      fold_vjt_1      |
|   fold_vjt_0   |  折叠神经元0的膜电平  |   fold_vjt_0   |      fold_vjt_0      |

#### 在线核神经元

**全神经元**与**半神经元**共享参数：

|    参数名（手册）    |          含义          |    参数模型检验名    | 参数模型导出名(json) |
| :------------------: | :--------------------: | :------------------: | :------------------: |
|    tick_relative    |        同离线核        |    tick_relative    |    tick_relative    |
|      addr_axon      |        同离线核        |      addr_axon      |      addr_axon      |
|     addr_core_xy     |        同离线核        |     addr_core_xy     |     addr_core_xy     |
|     addr_core_x     |        同离线核        |     addr_core_x     |     addr_core_x     |
|     addr_core_y     |        同离线核        |     addr_core_y     |     addr_core_y     |
|     addr_copy_xy     |        同离线核        |     addr_copy_xy     |     addr_copy_xy     |
|     addr_copy_x     |        同离线核        |     addr_copy_x     |     addr_copy_x     |
|     addr_copy_y     |        同离线核        |     addr_copy_y     |     addr_copy_y     |
|     weight_skew     |        同离线核        |     weight_skew     |     weight_skew     |
| weight_address_start |        同离线核        | weight_address_start | weight_address_start |
|  weight_address_end  |        同离线核        |  weight_address_end  |  weight_address_end  |
|     output_type     |    在线输出类型选择    |     output_type     |     output_type     |
|      fold_type      |        同离线核        |      fold_type      |      fold_type      |
|     neuron_type     |        同离线核        |     neuron_type     |     neuron_type     |
|         vjt         | 当前时间步膜电平(fp32) |         vjt         |         vjt         |

**全神经元**参数：

|   参数名（手册）   |      含义      |   参数模型检验名   | 参数模型导出名(json) |
| :-----------------: | :------------: | :-----------------: | :------------------: |
|     reset_mode     |    同离线核    |     reset_mode     |      reset_mode      |
|       reset_v       | 同离线核(bf16) |       reset_v       |       reset_v       |
| threshold_neg_mode |    同离线核    | threshold_neg_mode |  threshold_neg_mode  |
| threshold_pos_mode |    同离线核    | threshold_pos_mode |  threshold_pos_mode  |
|    threshold_neg    | 同离线核(fp32) |    threshold_neg    |    threshold_neg    |
|    threshold_pos    | 同离线核(fp32) |    threshold_pos    |    threshold_pos    |
| lateral_inhibition |    同离线核    | lateral_inhibition |  lateral_inhibition  |
| leak_multi_sequence |    同离线核    | leak_multi_sequence | leak_multi_sequence |
|  leak_multi_input  |    同离线核    |  leak_multi_input  |   leak_multi_input   |
|   leak_multi_mode   |    同离线核    |   leak_multi_mode   |   leak_multi_mode   |
|    leak_add_mode    |    同离线核    |    leak_add_mode    |    leak_add_mode    |
|      leak_tau      |    同离线核    |      leak_tau      |       leak_tau       |
|       leak_v       | 同离线核(bf16) |       leak_v       |        leak_v        |
|   weight_compress   |    同离线核    |   weight_compress   |   weight_compress   |
|     vjt_initial     | 同离线核(bf16) |     vjt_initial     |     vjt_initial     |

**折叠神经元**参数：

| 参数名（手册） |      含义      | 参数模型检验名 | 参数模型导出名(json) |
| :------------: | :------------: | :------------: | :------------------: |
| fold_range_xy |    同离线核    | fold_range_xy |    fold_range_xy    |
|  fold_range_x  |    同离线核    |  fold_range_x  |     fold_range_x     |
|  fold_range_y  |    同离线核    |  fold_range_y  |     fold_range_y     |
|  fold_skew_xy  |    同离线核    |  fold_skew_xy  |     fold_skew_xy     |
|  fold_skew_x  |    同离线核    |  fold_skew_x  |     fold_skew_x     |
|  fold_skew_y  |    同离线核    |  fold_skew_y  |     fold_skew_y     |
|  fold_axon_xy  |    同离线核    |  fold_axon_xy  |     fold_axon_xy     |
|  fold_axon_x  |    同离线核    |  fold_axon_x  |     fold_axon_x     |
|  fold_axon_y  |    同离线核    |  fold_axon_y  |     fold_axon_y     |
|  fold_number  |    同离线核    |  fold_number  |     fold_number     |
|   fold_vjt_3   | 同离线核(fp32) |   fold_vjt_3   |      fold_vjt_3      |
|   fold_vjt_2   | 同离线核(fp32) |   fold_vjt_2   |      fold_vjt_2      |
|   fold_vjt_1   | 同离线核(fp32) |   fold_vjt_1   |      fold_vjt_1      |
|   fold_vjt_0   | 同离线核(fp32) |   fold_vjt_0   |      fold_vjt_0      |
