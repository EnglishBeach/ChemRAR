from __future__ import annotations

import threading

from aizynthfinder import aizynthfinder
from pydantic import BaseModel
from rdkit import Chem as rd

from . import config, utils

_LOCK = threading.Lock()


class SearchConfig(config.AlgorithmConfig):
    algorithm: config.Algorithm


class _EngineMixin:
    _engine: aizynthfinder.AiZynthFinder

    @property
    def expansion_policies(self) -> list[str]:
        return list(self._engine.expansion_policy.items)

    @property
    def filters(self) -> list[str]:
        return list(self._engine.filter_policy.items)

    @property
    def stocks(self) -> list[str]:
        return list(self._engine.stock.items)

    @property
    def scorers(self) -> list[str]:
        return list(self._engine.scorers.names())

    @property
    def search_config(self) -> SearchConfig:
        full_config = self._engine.config.search.algorithm_config | dict(
            algorithm=self._engine.config.search.algorithm
        )
        return SearchConfig(**full_config)  # type: ignore

    def _get_engine(self):
        return _copy_engine(self._engine)


class Builder(_EngineMixin):
    def __init__(
        self,
        stock: dict[str, config.Stock],
        expansion: dict[str, config.ExpansionPolicy],
        search: config.Search | None = None,
        post_processing: config.PostProcessing | None = None,
        filter: dict[str, config.FilterPolicy] | None = None,
    ) -> None:
        init_config = _InitConfig(
            search=search or config.Search(),
            post_processing=post_processing or config.PostProcessing(),
            expansion=expansion,
            filter=filter or {},
            stock=stock,
        )
        self._engine = aizynthfinder.AiZynthFinder(configdict=init_config.model_dump(mode="json"))
        self._engine.stock.select_all()
        self._engine.expansion_policy.select_all()
        self._engine.filter_policy.select_all()

    def generate_tree(  # noqa: PLR0913
        self,
        smiles: str,
        *,
        max_transforms: int | None = None,
        time_limit: int | None = None,
        iteration_limit: int | None = None,
        return_first: bool | None = None,
        search_rewards: list[str] | None = None,
    ) -> Tree:
        if not rd.MolFromSmiles(smiles):
            msg = f"SMILES: {smiles} is incorrect"
            raise ValueError(msg)

        engine = self._get_engine()
        _change_search_configs(
            engine,
            max_transforms=max_transforms,
            time_limit=time_limit,
            iteration_limit=iteration_limit,
            return_first=return_first,
            search_rewards=search_rewards,
        )

        engine.target_smiles = smiles
        engine.prepare_tree()
        time = engine.tree_search(show_progress=False)
        return Tree(engine=engine, build_time=time)


class Tree(_EngineMixin):
    build_time: float

    def __init__(self, engine: aizynthfinder.AiZynthFinder, build_time: float) -> None:
        self._engine = engine
        self.build_time = build_time

    def continue_tree(
        self,
        iterations: int,
    ):
        engine = self._get_engine()
        _change_search_configs(
            engine,
            iteration_limit=engine.config.search.iteration_limit + iterations,
        )
        time = engine.tree_search(show_progress=False)
        return Tree(engine=engine, build_time=time + self.build_time)

    def score(self, scorers: list[str] | None = None):
        scorers = scorers or self.scorers


class _InitConfig(BaseModel):
    search: config.Search = config.Search()
    post_processing: config.PostProcessing = config.PostProcessing()
    expansion: dict[str, config.ExpansionPolicy] = {}
    filter: dict[str, config.FilterPolicy] = {}
    stock: dict[str, config.Stock] = {}


def _copy_engine(engine: aizynthfinder.AiZynthFinder) -> aizynthfinder.AiZynthFinder:
    with _LOCK:
        new = utils.copy_dataclass(engine)
        new.config = utils.copy_dataclass(new.config)

        # Mutate search.algorithm_config.search_rewards in: search
        new.config.search = utils.copy_dataclass(new.config.search)
        new.config.search.algorithm_config = utils.copy_dataclass(
            new.config.search.algorithm_config
        )

        # Mutate _items in: cached_search[stock.__contains__(mol)/.stock._apply_stop_criteria]
        stock = utils.copy_dataclass(new.stock)
        if stock._use_stop_criteria:  # noqa: SLF001
            stock._items = {k: utils.copy_dataclass(v) for k, v in stock._items.items()}  # noqa: SLF001
        new.stock = stock
        new.config.stock = stock

        # Mutate _items in: reset_cache[prepare_tree/.filter_policy.reset_cache]
        filter_policy = utils.copy_dataclass(new.filter_policy)
        filter_policy._items = {  # noqa: SLF001
            k: utils.copy_dataclass(v)
            for k, v in filter_policy._items.items()  # noqa: SLF001
        }
        new.filter_policy = filter_policy
        new.config.filter_policy = filter_policy

        # Mutate _items in: reset_cache[prepare_tree/.expansion_policy.reset_cache]
        expansion_policy = utils.copy_dataclass(new.expansion_policy)
        expansion_policy._items = {  # noqa: SLF001
            k: utils.copy_dataclass(v)
            for k, v in expansion_policy._items.items()  # noqa: SLF001
        }
        new.expansion_policy = expansion_policy
        new.config.expansion_policy = expansion_policy

        # Mutate _items in: search
        scorers = utils.copy_dataclass(new.scorers)
        scorers._items = {k: utils.copy_dataclass(v) for k, v in scorers._items.items()}  # noqa: SLF001
        new.scorers = scorers
        new.config.scorers = scorers

        # TODO: copy logger
        return new


def _change_search_configs(  # noqa: PLR0913, PLR0917
    engine: aizynthfinder.AiZynthFinder,
    max_transforms: int | None = None,
    time_limit: int | None = None,
    iteration_limit: int | None = None,
    return_first: bool | None = None,
    search_rewards: list[str] | None = None,
):
    if max_transforms is not None:
        engine.config.search.max_transforms = max_transforms

    if return_first is not None:
        engine.config.search.return_first = return_first

    if time_limit is not None:
        engine.config.search.time_limit = time_limit * 60

    if iteration_limit is not None:
        engine.config.search.iteration_limit = iteration_limit

    if search_rewards is not None:
        engine.config.search.algorithm_config["search_rewards"] = search_rewards
