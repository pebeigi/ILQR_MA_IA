from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Player:
    """Lightweight player metadata used by the game definition and solver."""
    index: int
    name: str
    state_dim: int
    control_dim: int
