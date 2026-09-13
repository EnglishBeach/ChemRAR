from __future__ import annotations

from enum import Enum
from pathlib import Path
import warnings

from aizynthfinder.context.config import Configuration  # noqa: F401
from aizynthfinder.context.scoring import Scorer as BaseScorer
from aizynthfinder.context.scoring.scorers_mols import MctsNode, ReactionTree  # noqa: F401
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


class Scaler(BaseModel):
    """Scaler parameters, need to normalize scores."""

    @property
    def up(self) -> bool: ...


# TODO: x=0.5 can be from 1 to 0 and clip set x=0
class MinMaxScaler(Scaler):
    min_val: float
    max_val: float
    reverse: bool
    scale_factor: float

    @property
    def up(self) -> bool:
        mark = (0.5 - int(self.reverse)) * self.scale_factor
        return mark < 0


class SquashScaler(Scaler):
    slope: float
    xoffset: float
    yoffset: float

    @property
    def up(self) -> bool:
        return self.slope > 0


class PowerScaler(Scaler):
    base_coefficient: float

    @property
    def up(self) -> bool:
        return True


class Score:
    """Score class.

    Range (in bracets - after rescaling if scaler is on):
    - min value (0) - hard, bad
    - max value (1) - easy, good

    This is up order for aizynthfinder,
    all internal scorers have _reverse order parameter inside if min-good, max-bad
    """

    scorer_type: type
    """Only BaseScorer types.

    Methods ._score_node and ._score_reaction_tree must score in up order:
    min - bad, max - good
    """

    up_order: bool = True
    """Mark up order (min - bad, max - good) or not"""

    def __init__(
        self,
        *,
        scaler: Scaler | None = None,
        **kwargs: dict,
    ) -> None:
        if not issubclass(self.scorer_type, BaseScorer):
            msg = f"{self.__class__.__name__}._scorer_type must be is subclass of BaseScorer"
            raise TypeError(msg)

        if (not self.up_order) and (not scaler or scaler.up):
            msg = "Score reversed - use scaler with up=False"
            warnings.warn(msg, stacklevel=2)

        self.scaler = scaler
        self.kwargs = kwargs

    def create_scorer(self, config: _utils.Configuration) -> BaseScorer:
        scorer: BaseScorer = self.scorer_type(
            config=config,
            scaler_params=self.scaler.model_dump() if self.scaler else None,
            **self.kwargs,
        )
        scorer._reverse_order = True  # noqa: SLF001
        return scorer
