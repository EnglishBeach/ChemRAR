from aizynthfinder.context import scoring as _scoring

from chemrar_retro import _utils, config as _config


class _MissingPrecursorsScorer(_config.BaseScorer):
    scorer_name = "missing precursors"

    def __init__(
        self,
        config: _utils.Configuration,
        scaler_params: _utils.StrDict | None = None,
    ) -> None:
        super().__init__(config, scaler_params)
        self._n_precursors_scorer = _scoring.NumberOfPrecursorsScorer(config)
        self._n_stock_precursors_scorer = _scoring.NumberOfPrecursorsInStockScorer(config=config)

    def __repr__(self) -> str:
        return self.scorer_name

    def _score_node(self, node: _utils.MctsNode) -> float:
        n_precursors: float = self._n_precursors_scorer(node)  # type: ignore
        stock_precursor: float = self._n_stock_precursors_scorer(node)  # type: ignore
        return n_precursors - stock_precursor

    def _score_reaction_tree(self, tree: _utils.ReactionTree) -> float:
        n_precursors: float = self._n_precursors_scorer(tree)  # type: ignore
        stock_precursor: float = self._n_stock_precursors_scorer(tree)  # type: ignore
        return n_precursors - stock_precursor


class Fraction(_config.Score):
    scorer_type = _scoring.FractionInStockScorer


class IntermediatesFraction(_config.Score):
    scorer_type = _scoring.FractionOfIntermediatesInStockScorer


class FractionSource(_config.Score):
    scorer_type = _scoring.FractionInSourceStockScorer


class NPrecursors(_config.Score):
    scorer_type = _scoring.NumberOfPrecursorsScorer
    up_order = False


class NStockPrecursors(_config.Score):
    scorer_type = _scoring.NumberOfPrecursorsInStockScorer


class NSourcePrecursors(_config.Score):
    scorer_type = _scoring.StockAvailabilityScorer


class MissingPrecursors(_config.Score):
    scorer_type = _MissingPrecursorsScorer
    up_order = False
