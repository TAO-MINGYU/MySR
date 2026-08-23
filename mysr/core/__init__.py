"""Core search API: configuration, orchestration, and results."""
from .config import SearchConfig
from .search import HallOfFame, fit

__all__ = ["SearchConfig", "HallOfFame", "fit"]
