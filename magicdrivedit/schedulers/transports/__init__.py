from .linear import Linear, Linear2
from .relinear import ReLinear
from .trigflow import TrigFlow
from .triglinear import TrigLinear

TRANSPORTS = {
    "Linear": Linear,
    "Linear2": Linear2,
    "ReLinear": ReLinear,
    "TrigFlow": TrigFlow,
    "TrigLinear": TrigLinear,
}
