# Table of Terms

|     V2.1参数名      |                   定义                   |   原类型名或变量名    | 输入参数模型的统一类型名或变量名 | 参数模型导出字典键 |
| :-----------------: | :--------------------------------------: | :-------------------: | :------------------------------: | :----------------: |
|  weight_det_stoch   |        突触整合确定或随机模式选择        |     LeakModeType      |     SynapticIntegrationMode      |    Same as V2.1    |
|   leak_det_stoch    |        泄露整合确定或随机模式选择        |    WeightModeType     |       LeakIntegrationMode        |    Same as V2.1    |
| leak_reversal_flag  |        泄露正向或反向反转模式选择        |   LeakDirectionType   |        LeakDirectionMode         |    Same as V2.1    |
|      leak_post      |           比较前后泄露模式选择           |  LeakComparisonType   |        LeakComparisonMode        |    Same as V2.1    |
|     reset_mode      |            膜电平复位模式选择            |     ResetModeType     |            ResetMode             |    Same as V2.1    |
|       reset_v       |          膜电平常数复位的复位值          |        reset_v        |             reset_v              |    Same as V2.1    |
| threshold_neg_mode  | 负阈值地板（饱和）或重置（复位）模式选择 | NegativeThresModeType |      NegativeThresholdMode       |    Same as V2.1    |
| threshold_mask_ctrl |              阈值随机数掩码              |    thres_mask_ctrl    |       threshold_mask_bits        |    Same as V2.1    |
|       vjt_pre       |               神经元膜电位               |        vjt_pre        |             vjt_init             |      vjt_init      |
