from __future__ import annotations

import collections.abc as collections
import copy
from enum import Enum
from pathlib import Path
import threading
import typing

from aizynthfinder import aizynthfinder as aizynth_api
from aizynthfinder.context import scoring as aizynth_scoring
from pydantic import BaseModel
from rdkit import Chem as rd

from . import config as retro_config

_LOCK = threading.Lock()
T = typing.TypeVar("T")


def mols_from_sdf(sdf_path: Path) -> list[rd.Mol]:
    """Extract RDKit molecules list from sdf file.

    :param sdf_path: Path to sdf file with molecules
    :return: List of RDKit molecules
    """
    sdf_system = rd.ForwardSDMolSupplier(
        sdf_path.as_posix(),
        # sanitize=True,
        removeHs=False,
    )

    molecules = []
    for sdf_mol in sdf_system:
        molecule = rd.Mol(sdf_mol)
        molecule.SetProp("source", sdf_path.as_posix())
        molecules.append(molecule)
    return molecules


class Engine(aizynth_api.AiZynthFinder):
    @property
    def search_rewards(self) -> dict[str, float]:
        rewards = self.config.search.algorithm_config["search_rewards"]
        if len(rewards) == 1:
            weights = [1]
        else:
            weights = self.config.search.algorithm_config["search_rewards_weights"]
        return dict(zip(rewards, weights, strict=True))


class ScalerType(Enum):
    Squash = "squash"
    MinMax = "min_max"
    Power = "power"


class Scaler(BaseModel):
    """Scaler parameters, need to normalize scores.

    Range:
    - 0 - easy, good
    - 1 - hard, bad
    """

    type: ScalerType
    min_val: int
    max_val: int
    reverse: bool


class Scorer:
    _scorer_type: type
    """Only Scorer types"""

    def __init__(
        self,
        *,
        scaler: Scaler,
        **kwargs: dict,
    ) -> None:
        self.scaler = scaler
        self.kwargs = kwargs

    def create_scorer(self, config: aizynth_api.Configuration) -> aizynth_scoring.Scorer:
        return self._scorer_type(
            config=config,
            scaler_params=self.scaler.model_dump(),
            **self.kwargs,
        )


def create_engine(
    *,
    stock: dict[str, retro_config.Stock],
    expansion: dict[str, retro_config.ExpansionPolicy],
    search: retro_config.Search | None = None,
    post_processing: retro_config.PostProcessing | None = None,
    filter: dict[str, retro_config.FilterPolicy] | None = None,
    scorers: list[tuple[type, dict]] | None = None,
) -> Engine:

    init_config = _InitConfig(
        search=search or retro_config.Search(),
        post_processing=post_processing or retro_config.PostProcessing(),
        expansion=expansion,
        filter=filter or {},
        stock=stock,
    )
    engine = Engine(configdict=init_config.model_dump(mode="json"))

    for scorer_type, scorer_parameters in scorers or []:
        engine.scorers.load(scorer_type(config=engine.config, **scorer_parameters))

    return engine


def select(
    engine: Engine,
    *,
    stocks: list[str] | None = None,
    expansion_policies: list[str] | None = None,
    filter_policies: list[str] | None = None,
    scorers: list[str] | None = None,
    search_scorers: dict[str, float | None] | None = None,
) -> Engine:
    e = _copy_engine(engine)

    if stocks:
        e.stock.select(stocks)

    if expansion_policies:
        e.expansion_policy.select(expansion_policies)

    if filter_policies:
        e.filter_policy.select(filter_policies)

    if scorers:
        e.scorers.select(scorers)

    # TODO: sum = 1?
    if search_scorers:
        if set(search_scorers) - set(e.scorers.names()):
            msg = "Only exists scorers are allowed"
            raise ValueError(msg)
        if (len(search_scorers) == 1) or all(i is None for i in search_scorers.values()):
            weights = []
        elif all(i is not None for i in search_scorers.values()):
            weights = list(search_scorers.values())
        else:
            msg = "Only this search scorers allowed: {}, {scorer:None} {scorer:<float>}"
            raise ValueError(msg)

        e.config.search.algorithm_config["search_rewards"] = list(search_scorers)
        e.config.search.algorithm_config["search_rewards_weights"] = weights

    return e


def generate_tree(
    engine: Engine,
    smiles: str,
    *,
    max_transforms: int | None = None,
    time_limit: int | None = None,
    return_first: bool | None = None,
    search_rewards: dict[str, float] | None = None,
) -> Engine:
    if not rd.MolFromSmiles(smiles):
        msg = f"SMILES: {smiles} is incorrect"
        raise ValueError(msg)

    engine = _copy_engine(engine)

    if not engine.config.expansion_policy.selection:
        msg = "No expansion policy selected"
        raise ValueError(msg)

    if not engine.config.search.algorithm_config["search_rewards"]:
        msg = "No search rewards selected"
        raise ValueError(msg)

    _change_search_configs(
        engine,
        max_transforms=max_transforms,
        time_limit=time_limit,
        iteration_limit=0,
        return_first=return_first,
        search_scorers=search_rewards,
    )

    engine.target_smiles = smiles
    engine.prepare_tree()
    return engine


def search_tree(engine: Engine, iterations: int):
    _change_search_configs(
        engine,
        iteration_limit=engine.config.search.iteration_limit + iterations,
    )
    engine.tree_search(show_progress=False)


class _InitConfig(BaseModel):
    search: retro_config.Search = retro_config.Search()
    post_processing: retro_config.PostProcessing = retro_config.PostProcessing()
    expansion: dict[str, retro_config.ExpansionPolicy] = {}
    filter: dict[str, retro_config.FilterPolicy] = {}
    stock: dict[str, retro_config.Stock] = {}


def _copy_engine(engine: Engine) -> Engine:
    with _LOCK:
        new = _copy_dataclass(engine)
        new.config = _copy_dataclass(new.config)

        # Mutate search.algorithm_config.search_rewards in: search
        new.config.search = _copy_dataclass(new.config.search)
        new.config.search.algorithm_config = _copy_dataclass(new.config.search.algorithm_config)

        # Mutate _items in: cached_search[stock.__contains__(mol)/.stock._apply_stop_criteria]
        stock = _copy_dataclass(new.stock)
        if stock._use_stop_criteria:  # noqa: SLF001
            stock._items = {k: _copy_dataclass(v) for k, v in stock._items.items()}  # noqa: SLF001
        new.stock = stock
        new.config.stock = stock

        # Mutate _items in: reset_cache[prepare_tree/.filter_policy.reset_cache]
        filter_policy = _copy_dataclass(new.filter_policy)
        filter_policy._items = {  # noqa: SLF001
            k: _copy_dataclass(v)
            for k, v in filter_policy._items.items()  # noqa: SLF001
        }
        new.filter_policy = filter_policy
        new.config.filter_policy = filter_policy

        # Mutate _items in: reset_cache[prepare_tree/.expansion_policy.reset_cache]
        expansion_policy = _copy_dataclass(new.expansion_policy)
        expansion_policy._items = {  # noqa: SLF001
            k: _copy_dataclass(v)
            for k, v in expansion_policy._items.items()  # noqa: SLF001
        }
        new.expansion_policy = expansion_policy
        new.config.expansion_policy = expansion_policy

        # Mutate _items in: search
        scorers = _copy_dataclass(new.scorers)
        scorers._items = {k: _copy_dataclass(v) for k, v in scorers._items.items()}  # noqa: SLF001
        new.scorers = scorers
        new.config.scorers = scorers

        # TODO: copy logger
        return new


def _copy_dataclass(obj: T) -> T:
    new_obj = copy.copy(obj)

    if hasattr(new_obj, "__dict__"):
        for attr, value in list(new_obj.__dict__.items()):
            if isinstance(value, collections.Collection):
                new_obj.__dict__[attr] = copy.copy(value)

    return new_obj


def _change_search_configs(
    engine: Engine,
    *,
    max_transforms: int | None = None,
    time_limit: int | None = None,
    iteration_limit: int | None = None,
    return_first: bool | None = None,
    search_scorers: dict[str, float] | None = None,
):
    if max_transforms is not None:
        engine.config.search.max_transforms = max_transforms

    if return_first is not None:
        engine.config.search.return_first = return_first

    if time_limit is not None:
        engine.config.search.time_limit = time_limit * 60

    if iteration_limit is not None:
        engine.config.search.iteration_limit = iteration_limit

    if search_scorers is not None:
        if set(search_scorers) - set(engine.scorers.names()):
            msg = "Only exists scorers are allowed"
            raise ValueError(msg)

        engine.config.search.algorithm_config["search_rewards"] = list(search_scorers)
        engine.config.search.algorithm_config["search_rewards_weights"] = list(
            search_scorers.values()
        )
