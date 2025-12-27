# Make yolo module accessible
from . import yolo

# For backward compatibility and easier imports
from .yolo import Model

__all__ = ['Model', 'yolo']