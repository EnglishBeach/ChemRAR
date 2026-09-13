from aizynthfinder.context import scoring as _scoring

from chemrar_retro import config as _config


class RouteCost(_config.Score):
    scorer_type = _scoring.RouteCostScorer
    up_order = False


class PrecursorsCost(_config.Score):
    scorer_type = _scoring.PriceSumScorer
    up_order = False
