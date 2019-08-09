REGISTRY = {}

from .basic_controller import BasicMAC
from .com_controller import ComMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["com_mac"] = ComMAC