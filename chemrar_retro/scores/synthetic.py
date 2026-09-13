from aizynthfinder.context import scoring as _scoring
import BRSAScore as br_sascore
import numpy as np

from chemrar_retro import _utils, parameters as _config


class _BRSAScore(_config.BaseScorer):
    scorer_name = "target br-sascore"

    def __init__(
        self,
        config: _utils.Configuration,
    ) -> None:
        super().__init__(config)
        self._model = br_sascore.SAScorer()
        self._cache: dict[str, float] = {}
        self._max_cache_len = 1000
        self._min_val = 1
        self._max_val = 10

    def __repr__(self) -> str:
        return self.scorer_name

    def _score_node(self, node: _utils.MctsNode) -> float:
        mean_score = np.mean(
            [
                self._calculate_score(i.smiles) if i.smiles else self._min_val
                for i in node.state.mols
            ]
        )
        return float(mean_score)

    def _score_reaction_tree(self, tree: _utils.ReactionTree) -> float:
        leaves = list(tree.leafs())
        mean_score = np.mean(
            [self._calculate_score(i.smiles) if i.smiles else self._min_val for i in leaves]
        )
        return float(mean_score)

    def _calculate_score(self, smiles: str) -> float:
        cached = self._cache.get(smiles, None)

        if cached is None:
            sascore, _ = self._model.calculateScore(smiles)
            sascore = (sascore - 1) / (self._max_val - 1)
            if len(self._cache) > self._max_cache_len:
                self._cache.popitem()

            self._cache[smiles] = sascore
            return sascore

        return self._cache[smiles]


class SCScore(_config.Score):
    scorer_type = _scoring.DeltaSyntheticComplexityScorer


class NPrecursors(_config.Score):
    scorer_type = _scoring.NumberOfPrecursorsScorer
    up_order = False


class BRSAScore(_config.Score):
    scorer_type = _BRSAScore
