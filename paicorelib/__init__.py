# isort: skip_file

# Coordinate definitions
from .coordinate import *

# Frame library
from .framelib import *

# Hardware constants
from .hw_defs import HwParams as HwConfig  # keep the compatibility
from .hw_defs import HwOfflineCoreParams as OffCoreCfg
from .hw_defs import HwOnlineCoreParams as OnCoreCfg

HwCfg = HwConfig

# Neuron RAM type definitions & model
from .ram_defs import LeakComparisonMode as LCM
from .ram_defs import LeakDirectionMode as LDM
from .ram_defs import LeakIntegrationMode as LIM
from .ram_defs import NegativeThresholdMode as NTM
from .ram_defs import ResetMode as RM
from .ram_defs import SynapticIntegrationMode as SIM
from .ram_defs import RAMDefs
from .ram_defs import OfflineRAMDefs as OffRAMDefs
from .ram_defs import OnlineRAMDefs as OnRAMDefs
from .ram_defs import OnlineRAMDefs_WW1 as OnRAMDefs_WW1
from .ram_defs import OnlineRAMDefs_WWn as OnRAMDefs_WWn
from .ram_model import *

# Core registers type definitions & model
from .reg_defs import *
from .reg_defs import RegDefs
from .reg_defs import OfflineRegDefs as OffRegDefs
from .reg_defs import OnlineRegDefs as OnRegDefs
from .reg_model import *

# Routing type definitions
from .routing_defs import *

# Version
from importlib.metadata import version

try:
    __version__ = version("paicorelib")
except Exception:
    __version__ = None
