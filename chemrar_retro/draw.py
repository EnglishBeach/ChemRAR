import math
from pathlib import Path

import networkx as nx
from PIL import (
    Image as pimage,
    ImageChops as pchops,
    ImageDraw as pdraw,
    ImageFont as pfont,
    ImageOps as pops,
)
from rdkit.Chem import Draw as rd_draw  # type: ignore

from . import retro


def draw_routes_table(routes: list[retro.RouteStatistic]):
    scores: list[dict[str, float]] = [i.score for i in routes]
    images: list[pimage.Image] = [i.image for i in routes]

    score_keys = list(routes[0].score)

    titles = []
    for i, score_data in enumerate(scores):
        title = "\n".join(f"{score_data[key]:>4.1%}" for key in score_keys)
        titles.append(f"Rate: {i}\n{title}")

    cell_dx, cell_dy = (max(i.width for i in images), max(i.height for i in images))
    processed_images = [
        _add_border(
            _add_title(
                _pad_to_size(
                    image,
                    size=(cell_dx, cell_dy),
                ),
                title,
                font_size=50,
                text_color=(0, 0, 0),
                overlay_color=(0, 0, 0, 0),
            ),
        )
        for image, title in zip(images, titles, strict=True)
    ]
    return _combine_images_to_grid(processed_images, cols=2)


def draw_molecule_tree(
    g: nx.DiGraph,
    out_path: str,
    tmp_dir: Path = Path("tmp"),
    sub_img_size: tuple = (450, 450),
) -> None:
    import pydot  # type: ignore # noqa: PLC0415

    tmp_dir = tmp_dir.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dot = pydot.Dot(graph_type="digraph")
    dot.set("dpi", "300")
    ids = {node: f"n{i}" for i, node in enumerate(g.nodes)}

    for node, node_id in ids.items():
        mols = [i.rd_mol for i in node._state.mols]  # noqa: SLF001
        n = max(len(mols), 1)

        img = rd_draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=sub_img_size, returnPNG=False)
        img = _trim_whitespace(img)

        img_path = tmp_dir / f"{node_id}.png"
        img.save(img_path)

        # w_in = img.width / 300
        # h_in = img.height / 300

        dot.add_node(
            pydot.Node(
                node_id,
                shape="box",
                label="",
                image=img_path,
                # imagescale="true",
                # width=str(w_in),
                # height=str(h_in),
                # fixedsize="true",
            )
        )

    for u, v in g.edges:
        dot.add_edge(pydot.Edge(ids[u], ids[v]))
    dot.write_png(out_path)


def _pad_to_size(
    img: pimage.Image,
    *,
    size: tuple[int, int],
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> pimage.Image:
    return pops.pad(img, size, color=bg_color)


def _add_border(
    img: pimage.Image,
    *,
    width: int = 3,
    color: tuple[int, int, int] = (0, 0, 0),
) -> pimage.Image:
    return pops.expand(img, border=width, fill=color)


def _add_title(
    img: pimage.Image,
    title: str,
    *,
    font_size: int = 16,
    text_color: tuple[int, int, int] = (255, 255, 255),
    overlay_color: tuple[int, int, int, int] = (0, 0, 0, 120),
) -> pimage.Image:
    img = img.convert("RGBA")
    overlay = pimage.new("RGBA", img.size, (0, 0, 0, 0))
    draw = pdraw.Draw(overlay)
    font = pfont.load_default(size=font_size)
    bar_h = font_size + 12
    draw.rectangle([0, 0, img.width, bar_h], fill=overlay_color)
    # bbox = draw.textbbox((0, 0), title, font=font)
    # text_w = bbox[2] - bbox[0]
    # text_x = (img.width - text_w) // 2
    text_x = 10
    draw.text((text_x, 6), title, fill=text_color, font=font)
    return pimage.alpha_composite(img, overlay).convert("RGB")


def _combine_images_to_grid(
    images: list[pimage.Image],
    *,
    cols: int | None = None,
    padding: int = 10,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> pimage.Image:
    n = len(images)
    cols = cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    total_w = cols * cell_w + (cols + 1) * padding
    total_h = rows * cell_h + (rows + 1) * padding
    grid = pimage.new("RGB", (total_w, total_h), bg_color)
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        x = padding + col * (cell_w + padding) + (cell_w - img.width) // 2
        y = padding + row * (cell_h + padding) + (cell_h - img.height) // 2
        grid.paste(img, (x, y))
    return grid


def _trim_whitespace(img: pimage.Image, bg: tuple = (255, 255, 255)) -> pimage.Image:
    bg_img = pimage.new(img.mode, img.size, bg)
    diff = pimage.eval(pimage.blend(img, bg_img, 0.0).convert("L"), lambda x: 255 - x)

    diff = pchops.difference(img.convert("RGB"), bg_img.convert("RGB"))
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img
