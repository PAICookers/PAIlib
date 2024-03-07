from .coordinate import *
from .framelib import *

# In recent versions, `HwConfig` for external packages remain unchanged.
from .hw_defs import HwParams as HwConfig
from .hw_types import *
from .ram_model import *
from .ram_types import LeakComparisonMode as LCM
from .ram_types import LeakDirectionMode as LDM
from .ram_types import LeakIntegrationMode as LIM
from .ram_types import NegativeThresholdMode as NTM
from .ram_types import ResetMode as RM
from .ram_types import SynapticIntegrationMode as SIM
from .ram_types import ThresholdMode as TM
from .reg_model import *
from .reg_types import CoreMode as CoreMode
from .reg_types import CoreModeDict as CoreModeDict
from .reg_types import InputWidthFormatType as InputWidthFormat
from .reg_types import LCNExtensionType as LCN_EX
from .reg_types import MaxPoolingEnableType as MaxPoolingEnable
from .reg_types import SNNModeEnableType as SNNModeEnable
from .reg_types import SpikeWidthFormatType as SpikeWidthFormat
from .reg_types import WeightPrecisionType as WeightPrecision
from .routing_defs import *


from importlib.metadata import version

try:
    __version__ = version("paicorelib")
except Exception:
    __version__ = None
