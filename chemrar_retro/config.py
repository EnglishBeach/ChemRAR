from __future__ import annotations

from enum import Enum
from pathlib import Path

from aizynthfinder.context.scoring import Scorer as BaseScorer
from pydantic import BaseModel

from chemrar_retro import _utils


# Search
class Algorithm(Enum):
    MCTS = "mcts"


class MCTSGrouping(Enum):
    partial = "partial"
    full = "full"


class AlgorithmConfig(BaseModel):
    C: float = 1.4
    default_prior: float = 0.5
    use_prior: bool = True
    prune_cycles_in_search: bool = True

    search_rewards: list[str] = ["state score"]
    search_rewards_weights: list[float] = []
    # TODO: only exists expansion pols
    immediate_instantiation: list[str] = []
    mcts_grouping: MCTSGrouping | None = None


class BreakBondsOperator(Enum):
    And = "and"
    Or = "or"


class Search(BaseModel):
    # TODO: custom algorithm
    algorithm: Algorithm = Algorithm.MCTS
    algorithm_config: AlgorithmConfig = AlgorithmConfig()

    max_transforms: int = 6
    iteration_limit: int = 100
    time_limit: int = 120
    return_first: bool = False
    exclude_target_from_stock: bool = True
    break_bonds: list[list[int]] = []
    freeze_bonds: list[list[int]] = []
    break_bonds_operator: BreakBondsOperator = BreakBondsOperator.And


# Expansion
# TODO: Does it use in non template-based?
class ExpansionPolicy(BaseModel):
    model: Path
    template: Path
    type: str = "template-based"
    template_column: str = "retro_template"
    cutoff_cumulative: float = 0.995
    cutoff_number: int = 50
    use_rdchiral: bool = True
    use_remote_models: bool = False
    rescale_prior: bool = False
    mask: Path | None = None


# Filter
# TODO: Does it use in non quick-filter?
class FilterPolicy(BaseModel):
    model: Path
    type: str = "quick-filter"
    exclude_from_policy: list[str] = []
    filter_cutoff: float = 0.05
    use_remote_models: bool = False


# Stock
class Stock(BaseModel):
    path: Path
    type: str = "inchiset"


# Score
class PostProcessing(BaseModel):
    min_routes: int = 5
    max_routes: int = 25
    all_routes: bool = False
    route_distance_model: Path | None = None
    route_scorers: list[str] = ["state score"]
    scorer_weights: list[float] | None = None


class ScalerType(Enum):
    Squash = "squash"
    MinMax = "min_max"
    Power = "power"


class Scaler(BaseModel):
    """Scaler parameters, need to normalize scores."""

    type: ScalerType = ScalerType.MinMax
    min_val: int = 0
    max_val: int = 1
    reverse: bool = False


class Score:
    """Score class.

    Range (in bracets - after rescaling if scaler is on):
    - min value (0) - hard, bad
    - max value (1)- easy, good

    This is straight order for aizynthfinder,
    all internal scorers have _reverse order parameter inside
    """

    _scorer_type: type
    """Only BaseScorer types"""

    straight_order: bool = True

    def __init__(
        self,
        *,
        scaler: Scaler | None = None,
        **kwargs: dict,
    ) -> None:
        if not issubclass(self._scorer_type, BaseScorer):
            msg = f"{self.__class__.__name__}._scorer_type must be is subclass of BaseScorer"
            raise TypeError(msg)

        # TODO: scale
        self.scaler = scaler
        self.kwargs = kwargs

    def create_scorer(self, config: _utils.Configuration) -> BaseScorer:
        return self._scorer_type(
            config=config,
            scaler_params=self.scaler.model_dump() if self.scaler else None,
            **self.kwargs,
        )
