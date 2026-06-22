from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

try:
    from ..game.player import Player
    from ..costs.player_cost import PlayerCost
    from ..dynamics.base_dynamics import ConcatenatedDynamics
except ImportError:  # pragma: no cover
    from game.player import Player
    from costs.player_cost import PlayerCost
    from dynamics.base_dynamics import ConcatenatedDynamics

@dataclass
class GameDefinition:
    """Container describing one finite-horizon ILQ game instance.

    This plays the role of the C++ ``Problem`` abstraction in
    ilqgames-master, but in a compact Python form.
    """
    players: List[Player]
    dynamics: ConcatenatedDynamics
    player_costs: List[PlayerCost]
    x0: np.ndarray
    dt: float
    horizon_steps: int
    name: str = "ILQ Game"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.x0 = np.asarray(self.x0, dtype=float).reshape(-1)
        if len(self.players) != len(self.player_costs):
            raise ValueError("players and player_costs must have the same length")
        if len(self.players) != self.dynamics.num_players:
            raise ValueError("number of players must match dynamics.num_players")

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def state_dim(self) -> int:
        return int(self.x0.size)

    @property
    def control_dims(self) -> list[int]:
        return [p.control_dim for p in self.players]

