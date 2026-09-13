# ruff: disable[F401]
import typing

from aizynthfinder.context import config as aizynth_config, scoring as aizynth_scoring
from aizynthfinder.context.scoring import scorers_mols as aizynth_scorers_mols
from aizynthfinder.context.scoring.scorers_mols import (
    DeltaSyntheticComplexityScorer as SCScore,
    # NumberOfPrecursorsScorer as NPrecursors,
)
from aizynthfinder.utils import type_utils as aizynth_types
import BRSAScore as br_sascore
import numpy as np

from . import _utils


class NPrecursors(aizynth_scorers_mols.NumberOfPrecursorsScorer):
    def __init__(
        self,
        config: aizynth_config.Configuration | None = None,
        scaler_params: dict[str, typing.Any] | None = None,
    ) -> None:
        if scaler_params:
            # TODO: log
            print("User's scaler_params skipped")  # noqa: T201

        scaler_params_ = _utils.ScalerParams(
            type=_utils.ScalerType.MinMax,
            min_val=1,
            max_val=10,
            reverse=False,
        )
        super().__init__(config, scaler_params_.model_dump())


class BRSAScore(aizynth_scoring.Scorer):
    scorer_name = "target br-sascore"

    def __init__(
        self,
        config: aizynth_config.Configuration,
    ) -> None:
        scaler_params = _utils.ScalerParams(
            type=_utils.ScalerType.MinMax,
            min_val=1,
            max_val=10,
            reverse=False,
        )
        super().__init__(config, scaler_params.model_dump())
        self._model = br_sascore.SAScorer()
        self._cache: dict[str, float] = {}
        self._max_cache_len = 1000
        self._none_val = scaler_params.max_val

    def __repr__(self) -> str:
        return self.scorer_name

    def _score_node(self, node: aizynth_scorers_mols.MctsNode) -> float:
        mean_score = np.mean(
            [
                self._calculate_score(i.smiles) if i.smiles else self._none_val
                for i in node.state.mols
            ]
        )
        return float(mean_score)

    def _score_reaction_tree(self, tree: aizynth_scorers_mols.ReactionTree) -> float:
        leaves = list(tree.leafs())
        mean_score = np.mean(
            [self._calculate_score(i.smiles) if i.smiles else self._none_val for i in leaves]
        )
        return float(mean_score)

    def _calculate_score(self, smiles: str) -> float:
        cached = self._cache.get(smiles, None)

        if cached is None:
            sascore, _ = self._model.calculateScore(smiles)

            if len(self._cache) > self._max_cache_len:
                self._cache.popitem()

            self._cache[smiles] = sascore
            return sascore

        return self._cache[smiles]
