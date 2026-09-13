from enum import Enum

from aizynthfinder import aizynthfinder as aizynth_api, reactiontree as aizynth_tree
from aizynthfinder.context import stock as aizynth_stock
from PIL import Image as pimage
from pydantic import BaseModel

from . import retro


class TreeType(Enum):
    MCTS = "MctsSearchTree"
    AndOr = "AndOrSearchTree"


class RouteInfo(BaseModel, arbitrary_types_allowed=True):
    score: dict[str, float]
    steps: int
    precursors: dict[str, str | None]
    solved: bool
    image: pimage.Image


class TreeInfo(BaseModel):
    tree_type: TreeType
    routes: list[RouteInfo]

    n_nodes: int
    max_transforms: int
    max_children: int
    n_solved: int

    steps: int
    policy_used_counts: dict


def analyze_tree(
    engine: retro.Engine,
    *,
    scorers: list[str] | None = None,
    top_n: int = 0,
) -> TreeInfo:
    # scorers = scorers or self.scorers
    top_n = top_n or 100
    selection = aizynth_api.RouteSelectionArguments(
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
        if isinstance(analysis_tree.search_tree, aizynth_api.MctsSearchTree)  # type: ignore
        else TreeType.AndOr
    )
    statistica = analysis_tree.tree_statistics()  # type: ignore

    return TreeInfo(
        tree_type=tree_type,
        n_nodes=statistica["number_of_nodes"],
        max_transforms=statistica["max_transforms"],
        max_children=statistica["max_children"],
        n_solved=statistica["number_of_solved_routes"],
        policy_used_counts=statistica["policy_used_counts"],
        steps=statistica["number_of_steps"],
        routes=routes,
    )


def _analyze_route(route: dict, stock: aizynth_stock.Stock) -> RouteInfo:
    tree: aizynth_tree.ReactionTree = route["reaction_tree"]

    precursors: dict[str, str | None] = {}
    for mol in tree.leafs():
        source = stock.availability_string(mol)
        source = None if source == "Not in stock" else source
        smiles: str = mol.smiles  # type: ignore
        precursors.update({smiles: source})

    return RouteInfo(
        score=route["score"],
        precursors=precursors,
        steps=len(list(tree.reactions())),
        solved=tree.is_solved,
        image=route["image"],
    )
