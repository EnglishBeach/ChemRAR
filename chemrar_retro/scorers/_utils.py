from enum import Enum

from aizynthfinder import aizynthfinder as aizynth_api, reactiontree as aizynth_tree
from aizynthfinder.context import scoring as aizynth_scoring
from pydantic import BaseModel


class ScalerType(Enum):
    Squash = "squash"
    MinMax = "min_max"
    Power = "power"


class Scaler(BaseModel):
    """Scaler parameters, need to normalize scores.

    Range:
    - 0 - easy, good
    - 1 - hard, bad
    """

    type: ScalerType
    min_val: int
    max_val: int
    reverse: bool


class Scorer:
    _scorer_type: type
    """Only Scorer types"""

    def __init__(
        self,
        *,
        scaler: Scaler,
        **kwargs: dict,
    ) -> None:
        self.scaler = scaler
        self.kwargs = kwargs

    def create_scorer(self, config: aizynth_api.Configuration) -> aizynth_scoring.Scorer:
        return self._scorer_type(
            config=config,
            scaler_params=self.scaler.model_dump(),
            **self.kwargs,
        )
