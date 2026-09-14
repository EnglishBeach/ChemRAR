from aizynthfinder.context import scoring as _scoring
from aizynthfinder.utils import sc_score
import BRSAScore as br_sascore
import numpy as np

from chemrar_retro import _utils, parameters as _config


class _BRSAScore(_config.BaseScorer):
    scorer_name = "br-sascore"

    def __init__(
        self,
        config: _utils.Configuration,
        scaler_params: _utils.StrDict | None = None,
        top: bool = True,
    ) -> None:
        super().__init__(config, scaler_params)
        self._model = br_sascore.SAScorer()
        self._cache: dict[str, float] = {}
        self._max_cache_len = 1000
        self._stock = config.stock
        self.top = top

    def __repr__(self) -> str:
        return self.scorer_name

    def _score_node(self, node: _utils.MctsNode) -> float:
        scores = [self._calculate_score(i.smiles) for i in node.state.mols]  # type: ignore
        return float(np.mean(scores))

    def _score_reaction_tree(self, tree: _utils.ReactionTree) -> float:
        if self.top:
            return self._calculate_score(tree.root.smiles)  # type: ignore

        scores = [self._calculate_score(i.smiles) for i in tree.leafs()]  # type: ignore
        return float(np.mean(scores))

    def _calculate_score(self, smiles: str) -> float:
        cached = self._cache.get(smiles, None)

        if cached is not None:
            return self._cache[smiles]

        sascore, _ = self._model.calculateScore(smiles)
        sascore = (10 - sascore) / (10 - 1)

        if len(self._cache) > self._max_cache_len:
            self._cache.popitem()

        self._cache[smiles] = sascore
        return sascore


class _PickleDeltaSyntheticComplexityScorer(_scoring.DeltaSyntheticComplexityScorer):
    scorer_name = "dsc score"

    def __init__(
        self,
        config: _utils.Configuration,
        sc_score_model: str,
        scaler_params: _utils.StrDict | None = None,
        horizon: int = 3,
    ) -> None:
        if scaler_params:
            msg = "Scaler injected in Scorer"
            raise ValueError(msg)

        scaler_params = {"name": "min_max", "min_val": -1.5, "max_val": 4, "reverse": False}
        _config.BaseScorer.__init__(self, config, scaler_params)
        self._config = config
        self.horizon = horizon
        self._model = sc_score.SCScore(sc_score_model)


class _PickleSyntheticComplexityScorer(_config.BaseScorer):
    scorer_name = "sc score"

    def __init__(
        self,
        config: _utils.Configuration,
        sc_score_model: str,
        scaler_params: _utils.StrDict | None = None,
    ) -> None:
        if scaler_params:
            msg = "Scaler injected in Scorer"
            raise ValueError(msg)

        scaler_params = {"name": "min_max", "min_val": -1.5, "max_val": 4, "reverse": False}
        super().__init__(config)
        self.scscorer = sc_score.SCScore(sc_score_model)

    def __repr__(self) -> str:
        return self.scorer_name

    def _score_node(self, node: _utils.MctsNode) -> float:
        return max(self.scscorer(mol.rd_mol) for mol in node.state.mols)

    def _score_reaction_tree(self, tree: _utils.ReactionTree) -> float:
        return self.scscorer(tree.root.rd_mol)


class DSCScore(_config.Score):
    scorer_type = _PickleDeltaSyntheticComplexityScorer


class SCScore(_config.Score):
    scorer_type = _PickleSyntheticComplexityScorer


class BRSAScore(_config.Score):
    scorer_type = _BRSAScore
