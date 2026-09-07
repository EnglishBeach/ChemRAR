from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel


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


class PostProcessing(BaseModel):
    min_routes: int = 5
    max_routes: int = 25
    all_routes: bool = False
    route_distance_model: Path | None = None
    route_scorers: list[str] = ["state score"]
    scorer_weights: list[float] | None = None


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


# TODO: Does it use in non quick-filter?
class FilterPolicy(BaseModel):
    model: Path
    type: str = "quick-filter"
    exclude_from_policy: list[str] = []
    filter_cutoff: float = 0.05
    use_remote_models: bool = False


class Stock(BaseModel):
    path: Path
    type: str = "inchiset"
