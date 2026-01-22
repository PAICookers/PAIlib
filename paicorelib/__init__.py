from importlib.metadata import version

# Coordinate definitions
from .coordinate import (
    Coord,
    OfflineCoord,
    OnlineCoord,
    CoordOffset,
    ReplicationId,
    CoordXY,
    CoordXYOffset,
    CoordZXYOffset,
    CoordXYUnitVec,
    ChipCoord,
    CoordAddr,
    CoordTuple2d,
    CoordTuple3d,
    CoordLike,
    RIdLike,
    CoordXYLike,
    CoordXYOffsetLike,
    CoordZXYOffsetLike,
    to_coord,
    to_coords,
    to_coordoffset,
    to_rid,
    to_coordxy,
    to_coordxys,
    to_coordxyoffset,
    to_coordzxyoffset,
)

# Chip v1
# Hardware constants
from .hw_defs import HwParams as HwConfig  # keep the compatibility
from .hw_defs import HwCoreParams as HwCoreCfg
from .hw_defs import HwOfflineCoreParams as OffCoreCfg
from .hw_defs import HwOnlineCoreParams as OnCoreCfg

HwCfg = HwConfig
# Chip v2.5
from .hw_defs import HwParamsV2 as HwConfigV2

# Frame library
from .framelib import (
    OfflineFrameGen,
    OnlineFrameGen,
    ChipFrameGen,
    FrameGenV2,
    OfflineFrameGenV2,
    OfflineConfigFrame1,
    OfflineConfigFrame2,
    OfflineConfigFrame3,
    OfflineConfigFrame4,
    OfflineWorkFrame1,
    OfflineWorkFrame2,
    OfflineWorkFrame3,
    OfflineWorkFrame4,
    OfflineTestInFrame1,
    OfflineTestInFrame2,
    OfflineTestInFrame3,
    OfflineTestInFrame4,
    OfflineTestOutFrame1,
    OfflineTestOutFrame2,
    OfflineTestOutFrame3,
    OfflineTestOutFrame4,
    OnlineConfigFrame1,
    OnlineConfigFrame2,
    OnlineConfigFrame3,
    OnlineConfigFrame4,
    OnlineWorkFrame1_1,
    OnlineWorkFrame1_2,
    OnlineWorkFrame1_3,
    OnlineWorkFrame1_4,
    OnlineWorkFrame2,
    OnlineWorkFrame3,
    OnlineWorkFrame4,
    OnlineTestInFrame1,
    OnlineTestInFrame2,
    OnlineTestInFrame3,
    OnlineTestInFrame4,
    OnlineTestOutFrame1,
    OnlineTestOutFrame2,
    OnlineTestOutFrame3,
    OnlineTestOutFrame4,
    FRAME_DTYPE,
    LUT_DTYPE,
    PAYLOAD_DATA_DTYPE,
    FrameArrayType,
    LUTDataType,
    PayloadDataType,
)

# Routing
from .routing_defs import (
    RoutingCoord,
    RoutingDirection,
    RoutingLevel,
    RoutingPath,
    RoutingStatus,
    get_replication_id,
    get_multicast_cores,
    ONLINE_CORES_BASE_COORD,
    ROUTING_DIRECTIONS_IDX,
)

# Core register definitions & parameters model
from .core_model import CoreReg, OfflineCoreReg, OnlineCoreReg
from .core_defs import (
    CoreRegLim,
    OfflineCoreRegLim,
    OnlineCoreRegLim,
    WeightWidth,
    LCN_EX,
    InputWidthFormat,
    SpikeWidthFormat,
    MaxPoolingEnable,
    SNNModeEnable,
    LUTRandomEnable,
    DecayRandomEnable,
    LeakOrder,
    OnlineModeEnable,
    CoreType,
    CoreMode,
    get_core_mode,
    core_mode_check,
)

# Neuron register definitions & parameters model
from .neuron_model import (
    NeuDestInfo,
    OfflineNeuDestInfo,
    OnlineNeuDestInfo,
    NeuAttrs,
    OfflineNeuAttrs,
    OnlineNeuAttrs,
    OfflineNeuConf,
    OnlineNeuConf,
)
from .neuron_defs import (
    NeuRegLim,
    OfflineNeuRegLim,
    OnlineNeuRegLim,
    OnlineNeuRegLim_WW1,
    OnlineNeuRegLim_WWn,
    SynapticIntegrationMode as SIM,
    ResetMode as RM,
    NegativeThresholdMode as NTM,
    LeakIntegrationMode as LIM,
    LeakDirectionMode as LDM,
    LeakComparisonMode as LCM,
)

# Chip v2.5
# Core register definitions & parameters model
from .core_model_v2 import OfflineCoreRegV2
from .core_defs_v2 import (
    AddPotentialMode,
    CSCAccelerateMode,
    DataSign,
    DataWidth,
    OfflineCoreRegLimV2,
    PoolingMode,
    SNNMode,
    ZeroOutputMode,
)

# Neuron register definitions & parameters model
from .neuron_model_v2 import (
    NeuDestInfoV2,
    OfflineNeuDestInfoV2,
    NeuCommonAttrsV2,
    OfflineNeuCommonAttrsV2,
    OfflineNeuHalfAttrsV2,
    OfflineNeuFullAttrsV2,
    OfflineNeuFoldedAttrsV2Part1,
    OfflineNeuFoldedAttrsV2Part2,
    OnlineNeuFoldedAttrsV2Part1,
    OnlineNeuFoldedAttrsV2Part2,
    OfflineNeuFullConfV2,
    OfflineNeuHalfConfV2,
)
from .neuron_defs_v2 import (
    OutputType,
    FoldType,
    NeuronType,
    ThresholdNegMode,
    ThresholdPosMode,
    LateralInhibitionMode,
    LeakMultiComparisonOrder,
    LeakMultiInputMode,
    LeakMultiMode,
    LeakAddMode,
    WeightCompressType,
    OfflineNeuRegLimV2,
)

# Routing
from .routing_hexa import (
    AERPacketZXYCopy,
    AERPacket,
    aer_packet_walk,
    aer_packet_area,
    find_coordxy_shortest_path,
)

try:
    __version__ = version("paicorelib")
except Exception:
    __version__ = None
