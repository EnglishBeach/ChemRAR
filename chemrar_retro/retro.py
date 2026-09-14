from __future__ import annotations

from pathlib import Path
import threading

from aizynthfinder import aizynthfinder as aizynth_api
from pydantic import BaseModel
from rdkit import Chem as rd

from . import _utils, parameters as _config

_LOCK = threading.Lock()


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


def mol_to_sdf(molecule: rd.Mol, path: Path, rewrite: bool = False):
    """Write mol to sdf file.

    :param molecule: RDKit molecule
    :param path: Path to save file .sdf
    :param rewrite: Clear existing file or append new molecule, defaults to False
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    rdmols_to_save = []
    if path.exists() and not rewrite:
        saved_system = rd.ForwardSDMolSupplier(
            path.as_posix(),
            sanitize=False,
            removeHs=False,
        )

        for saved_mol in saved_system:
            rdmols_to_save.append(saved_mol)

    rdmols_to_save.append(molecule)

    with rd.SDWriter(path.as_posix()) as sdf:
        for mol in rdmols_to_save:
            sdf.write(mol)


class Engine(aizynth_api.AiZynthFinder):
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
    stock: dict[str, _config.Stock],
    expansion_policies: dict[str, _config.ExpansionPolicy],
    search: _config.Search | None = None,
    post_processing: _config.PostProcessing | None = None,
    filters: dict[str, _config.FilterPolicy] | None = None,
    scores: list[_config.Score] | None = None,
) -> Engine:

    init_config = _InitConfig(
        search=search or _config.Search(),
        post_processing=post_processing or _config.PostProcessing(),
        expansion=expansion_policies,
        filter=filters or {},
        stock=stock,
    )
    engine = Engine(configdict=init_config.model_dump(mode="json"))

    scores = scores or []
    scorers = [i.create_scorer(config=engine.config) for i in scores]
    names = [repr(i) for i in scorers]
    if len(names) != len(set(names)):
        msg = f"Some scorers have same names:\n {names}"
        raise ValueError(msg)
    for scorer in scorers:
        engine.scorers.load(scorer)

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
        return_first=return_first,
        search_scorers=search_rewards,
    )

    engine.target_smiles = smiles
    engine.prepare_tree()
    return engine


def search_tree(engine: Engine, iterations: int | None = None):
    if iterations:
        _change_search_configs(
            engine,
            iteration_limit=engine.config.search.iteration_limit + iterations,
        )
    engine.tree_search(show_progress=False)


class _InitConfig(BaseModel):
    search: _config.Search = _config.Search()
    post_processing: _config.PostProcessing = _config.PostProcessing()
    expansion: dict[str, _config.ExpansionPolicy] = {}
    filter: dict[str, _config.FilterPolicy] = {}
    stock: dict[str, _config.Stock] = {}


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
