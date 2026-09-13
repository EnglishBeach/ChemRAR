from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.context.config import Configuration
from aizynthfinder.context.scoring.scorers import CombinedScorer
from aizynthfinder.context.scoring.scorers_base import Scorer
from aizynthfinder.context.scoring.scorers_reactions import (
    AverageTemplateOccurrenceScorer,
    NumberOfReactionsScorer,
)
from BRSAScore import SAScorer
from RAscore import RAscore_XGB
from rdkit import Chem


class GatedComplexityScorer(Scorer):
    scorer_name = "gated synthetic complexity"

    def __init__(
        self, config: Configuration, route_scorer: Scorer, fallback_scorer: Scorer
    ) -> None:
        super().__init__(config)
        self._route_scorer = route_scorer
        self._fallback_scorer = fallback_scorer

    def __repr__(self) -> str:
        return self.scorer_name

    def _score_node(self, node) -> float:
        if node.state.is_solved:
            return self._route_scorer(node)
        return self._fallback_scorer(node)

    def _score_reaction_tree(self, tree) -> float:
        if tree.is_solved:
            return self._route_scorer(tree)
        return self._fallback_scorer(tree)


class TargetRAScoreScorer(Scorer):
    scorer_name = "target ra score"

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._model = RAscore_XGB.RAScorerXGB()

    def __repr__(self) -> str:
        return self.scorer_name

    def _score_node(self, node) -> float:
        smiles = Chem.MolToSmiles(node.state.mols[0].rd_mol)
        return float(self._model.predict(smiles))

    def _score_reaction_tree(self, tree) -> float:
        return float(self._model.predict(tree.mol_smiles))


def make_complexity_scorer(config: Configuration) -> CombinedScorer:
    scorers = config.scorers
    scorers.load(
        NumberOfReactionsScorer(
            config,
            scaler_params={
                "type": "min_max",
                "min_val": 0,
                "max_val": 10,
                "reverse": False,
            },
        )
    )
    scorers.load(
        AverageTemplateOccurrenceScorer(
            config,
            scaler_params={
                "type": "min_max",
                "min_val": 0,
                "max_val": 100,
                "reverse": True,
            },
        )
    )
    scorers.load(
        MissingPrecursorsScorer(
            config,
            scaler_params={
                "type": "min_max",
                "min_val": 0,
                "max_val": 5,
                "reverse": False,
            },
        )
    )
    return CombinedScorer(
        config,
        scorers=[
            "number of reactions",
            "average template occurrence",
            "missing precursors",
        ],
        weights=[0.40, 0.20, 0.40],
        combine_strategy="mean-arithmetic",
        short_name="synthetic complexity",
    )


def make_gated_complexity_scorer(config: Configuration) -> GatedComplexityScorer:
    route_scorer = make_complexity_scorer(config)
    fallback_scorer = TargetBRSAScoreScorer(config)
    return GatedComplexityScorer(config, route_scorer, fallback_scorer)


def triage_unsolved(finder: AiZynthFinder, targets: list[str]) -> list[tuple[str, float]]:
    scorer = RAscore_XGB.RAScorerXGB()
    unsolved = []
    for smiles in targets:
        finder.target_smiles = smiles
        finder.tree_search()
        finder.build_routes()
        if not finder.tree.is_solved:
            unsolved.append((smiles, float(scorer.predict(smiles))))
    return sorted(unsolved, key=lambda pair: pair[1], reverse=True)
