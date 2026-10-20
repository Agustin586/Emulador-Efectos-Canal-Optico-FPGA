"""Funciones del error de temporización (golden model)."""
from .delay import cubic_delay, cubic_taps, delay_error_db, ideal_delay, linear_delay

__all__ = ["cubic_delay", "cubic_taps", "delay_error_db", "ideal_delay", "linear_delay"]
