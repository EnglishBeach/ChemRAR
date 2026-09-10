from __future__ import annotations

from enum import Enum
import threading

from aizynthfinder import aizynthfinder as zynth_api, reactiontree as zynth_tree
from aizynthfinder.context import stock as zynth_stock
from PIL import Image as pimage
from pydantic import BaseModel
from rdkit import Chem as rd

from . import _utils, config as retro_config

_LOCK = threading.Lock()


class Engine(zynth_api.AiZynthFinder):
    @property
    def search_rewards(self) -> dict[str, float]:
        rewards = self.config.search.algorithm_config["search_rewards"]
        if len(rewards) == 1:
            weights = [1]
        else:
            weights = self.config.search.algorithm_config["search_rewards_weights"]
        return dict(zip(rewards, weights, strict=True))


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
    search_scorers: dict[str, float] | None = None,
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
        if len(search_scorers):
            weights = []
        else:
            weights = list(search_scorers.values())
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


# TODO: copy tree
def search_tree(engine: Engine, iterations: int):
    _change_search_configs(
        engine,
        iteration_limit=engine.config.search.iteration_limit + iterations,
    )
    engine.tree_search(show_progress=False)


class TreeType(Enum):
    MCTS = "MctsSearchTree"
    AndOr = "AndOrSearchTree"


class RouteStatistic(BaseModel, arbitrary_types_allowed=True):
    score: dict[str, float]
    steps: int
    precursors: dict[str, str | None]
    solved: bool
    image: pimage.Image


class ScoringStatistics(BaseModel):
    tree_type: TreeType
    routes: list[RouteStatistic]

    n_nodes: int
    max_transforms: int
    max_children: int
    n_solved: int

    steps: int
    policy_used_counts: dict


def analyze_tree(
    engine: Engine,
    *,
    scorers: list[str] | None = None,
    top_n: int = 0,
):
    # scorers = scorers or self.scorers
    top_n = top_n or 100
    selection = zynth_api.RouteSelectionArguments(
        return_all=not bool(top_n),
        nmin=top_n,
        nmax=top_n,
    )
    engine.build_routes(scorer=scorers, selection=selection)

    engine.routes.make_images()
    routes = [_analyze_route(route=route, stock=engine.stock) for route in engine.routes]

    analysis_tree = engine.analysis
    tree_type = (
        TreeType.MCTS
        if isinstance(analysis_tree.search_tree, zynth_api.MctsSearchTree)  # type: ignore
        else TreeType.AndOr
    )
    statistica = analysis_tree.tree_statistics()  # type: ignore

    return ScoringStatistics(
        tree_type=tree_type,
        n_nodes=statistica["number_of_nodes"],
        max_transforms=statistica["max_transforms"],
        max_children=statistica["max_children"],
        n_solved=statistica["number_of_solved_routes"],
        policy_used_counts=statistica["policy_used_counts"],
        steps=statistica["number_of_steps"],
        routes=routes,
    )


class _InitConfig(BaseModel):
    search: retro_config.Search = retro_config.Search()
    post_processing: retro_config.PostProcessing = retro_config.PostProcessing()
    expansion: dict[str, retro_config.ExpansionPolicy] = {}
    filter: dict[str, retro_config.FilterPolicy] = {}
    stock: dict[str, retro_config.Stock] = {}


def _copy_engine(engine: Engine) -> Engine:
    with _LOCK:
        new = _utils.copy_dataclass(engine)
        new.config = _utils.copy_dataclass(new.config)

        # Mutate search.algorithm_config.search_rewards in: search
        new.config.search = _utils.copy_dataclass(new.config.search)
        new.config.search.algorithm_config = _utils.copy_dataclass(
            new.config.search.algorithm_config
        )

        # Mutate _items in: cached_search[stock.__contains__(mol)/.stock._apply_stop_criteria]
        stock = _utils.copy_dataclass(new.stock)
        if stock._use_stop_criteria:  # noqa: SLF001
            stock._items = {k: _utils.copy_dataclass(v) for k, v in stock._items.items()}  # noqa: SLF001
        new.stock = stock
        new.config.stock = stock

        # Mutate _items in: reset_cache[prepare_tree/.filter_policy.reset_cache]
        filter_policy = _utils.copy_dataclass(new.filter_policy)
        filter_policy._items = {  # noqa: SLF001
            k: _utils.copy_dataclass(v)
            for k, v in filter_policy._items.items()  # noqa: SLF001
        }
        new.filter_policy = filter_policy
        new.config.filter_policy = filter_policy

        # Mutate _items in: reset_cache[prepare_tree/.expansion_policy.reset_cache]
        expansion_policy = _utils.copy_dataclass(new.expansion_policy)
        expansion_policy._items = {  # noqa: SLF001
            k: _utils.copy_dataclass(v)
            for k, v in expansion_policy._items.items()  # noqa: SLF001
        }
        new.expansion_policy = expansion_policy
        new.config.expansion_policy = expansion_policy

        # Mutate _items in: search
        scorers = _utils.copy_dataclass(new.scorers)
        scorers._items = {k: _utils.copy_dataclass(v) for k, v in scorers._items.items()}  # noqa: SLF001
        new.scorers = scorers
        new.config.scorers = scorers

        # TODO: copy logger
        return new


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


def _analyze_route(route: dict, stock: zynth_stock.Stock) -> RouteStatistic:
    tree: zynth_tree.ReactionTree = route["reaction_tree"]

    precursors: dict[str, str | None] = {}
    for mol in tree.leafs():
        source = stock.availability_string(mol)
        source = None if source == "Not in stock" else source
        smiles: str = mol.smiles  # type: ignore
        precursors.update({smiles: source})

    return RouteStatistic(
        score=route["score"],
        precursors=precursors,
        steps=len(list(tree.reactions())),
        solved=tree.is_solved,
        image=route["image"],
    )
