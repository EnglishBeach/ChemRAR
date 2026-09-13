from aizynthfinder.context import scoring as _scoring

from chemrar_retro import config as _config


class Similarity(_config.Score):
    scorer_type = _scoring.RouteSimilarityScorer


class TemplateOccurrence(_config.Score):
    scorer_type = _scoring.AverageTemplateOccurrenceScorer


class MaxTransform(_config.Score):
    scorer_type = _scoring.MaxTransformScorer
    up_order = False


class NReactions(_config.Score):
    scorer_type = _scoring.NumberOfReactionsScorer
    up_order = False


class ClassReaction(_config.Score):
    scorer_type = _scoring.ReactionClassMembershipScorer


class ClassRankReaction(_config.Score):
    scorer_type = _scoring.ReactionClassRankScorer
