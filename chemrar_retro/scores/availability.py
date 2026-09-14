from aizynthfinder.context import scoring as _scoring

from chemrar_retro import parameters as _config


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
