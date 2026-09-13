# ruff: disable[F401]

from aizynthfinder.context import config as aizynth_config, scoring as aizynth_scoring
from aizynthfinder.context.scoring import scorers_mols as aizynth_scorers_mols
from aizynthfinder.context.scoring.scorers_mols import (
    FractionInSourceStockScorer as FractionSource,
    FractionInStockScorer as Fraction,
    FractionOfIntermediatesInStockScorer as IntermediatesFraction,
    NumberOfPrecursorsInStockScorer as NStockPrecursors,
    NumberOfPrecursorsScorer as NPrecursors,
    StockAvailabilityScorer as NSourcePrecursors,
)
from aizynthfinder.utils import type_utils as aizynth_types


class MissingPrecursors(aizynth_scoring.Scorer):
    scorer_name = "missing precursors"

    def __init__(
        self,
        config: aizynth_config.Configuration,
        scaler_params: aizynth_types.StrDict | None = None,
    ) -> None:
        super().__init__(config, scaler_params)
        self._n_precursors_scorer = NPrecursors(config)
        self._n_stock_precursors_scorer = NStockPrecursors(config=config)

    def __repr__(self) -> str:
        return self.scorer_name

    def _score_node(self, node: aizynth_scorers_mols.MctsNode) -> float:
        n_precursors: float = self._n_precursors_scorer(node)  # type: ignore
        stock_precursor: float = self._n_stock_precursors_scorer(node)  # type: ignore
        return n_precursors - stock_precursor

    def _score_reaction_tree(self, tree: aizynth_scorers_mols.ReactionTree) -> float:
        n_precursors: float = self._n_precursors_scorer(tree)  # type: ignore
        stock_precursor: float = self._n_stock_precursors_scorer(tree)  # type: ignore
        return n_precursors - stock_precursor
