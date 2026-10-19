"""EEG-based early warning for autistic meltdowns (research prototype, not a medical device)."""
from .config import Config
from .detector import Alert, Detector, Step

__all__ = ["Config", "Detector", "Alert", "Step"]
__version__ = "0.1.0"
