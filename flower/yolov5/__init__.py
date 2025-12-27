# YOLOv5 package initialization
try:
    from . import models
except ImportError as e:
    print(f"Warning: Could not import models: {e}")

__all__ = ['models']