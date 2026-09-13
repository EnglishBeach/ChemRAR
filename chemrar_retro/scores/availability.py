from aizynthfinder.context.scoring import scorers_mols

from chemrar_retro import _utils, config as _config


class _MissingPrecursorsScorer(_config.BaseScorer):
    scorer_name = "missing precursors"

    def __init__(
        self,
        config: _utils.Configuration,
        scaler_params: _utils.StrDict | None = None,
    ) -> None:
        super().__init__(config, scaler_params)
        self._n_precursors_scorer = scorers_mols.NumberOfPrecursorsScorer(config)
        self._n_stock_precursors_scorer = scorers_mols.NumberOfPrecursorsInStockScorer(
            config=config
        )

    def __repr__(self) -> str:
        return self.scorer_name

    def _score_node(self, node: scorers_mols.MctsNode) -> float:
        n_precursors: float = self._n_precursors_scorer(node)  # type: ignore
        stock_precursor: float = self._n_stock_precursors_scorer(node)  # type: ignore
        return n_precursors - stock_precursor

    def _score_reaction_tree(self, tree: scorers_mols.ReactionTree) -> float:
        n_precursors: float = self._n_precursors_scorer(tree)  # type: ignore
        stock_precursor: float = self._n_stock_precursors_scorer(tree)  # type: ignore
        return n_precursors - stock_precursor


class Fraction(_config.Score):
    _scorer_type: type = scorers_mols.FractionInStockScorer


class IntermediatesFraction(_config.Score):
    _scorer_type: type = scorers_mols.FractionOfIntermediatesInStockScorer


class FractionSource(_config.Score):
    _scorer_type: type = scorers_mols.FractionInSourceStockScorer


class NPrecursors(_config.Score):
    _scorer_type: type = scorers_mols.NumberOfPrecursorsScorer


class NStockPrecursors(_config.Score):
    _scorer_type: type = scorers_mols.NumberOfPrecursorsInStockScorer


class NSourcePrecursors(_config.Score):
    _scorer_type: type = scorers_mols.StockAvailabilityScorer


class MissingPrecursors(_config.Score):
    _scorer_type: type = _MissingPrecursorsScorer
