"""Floor plans of the random flats: one PNG per flat and a contact sheet.

    uv run --project <team repo> --group ml python homes/render.py            # first 24 flats + sheet.png
    uv run --project <team repo> --group ml python homes/render.py --all      # every flat in flats.json

Reads data/flats.json (written by generate.py) and writes data/maps/.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
COLOURS = {"bedroom": "#cfdcf2", "living_room": "#f3d3cd", "dining_area": "#f2e3c2", "kitchen": "#d3ead3",
           "hallway": "#e6e6e6"}
FURNITURE_COLOUR = "#8d7b68"


def draw(ax, flat, title=True, small=False, probs=None, order=None, truth=None):
    """The floor plan. `probs` {zone: p} shades the zones by probability and prints it; `order` (zone names) draws
    the search path from the dock, numbered; `truth` (a zone name) marks where she really is with a star."""
    w, d = flat["width"], flat["depth"]
    ax.add_patch(Rectangle((-w / 2, -d / 2), w, d, fill=True, fc="#f6f3ee", ec="#444", lw=2, zorder=1))
    top = max(probs.values()) if probs else 1.0
    for z in flat["zones"]:
        x0, y0 = z["x"] - z["size_x"] / 2, z["y"] - z["size_y"] / 2
        colour = COLOURS[z["name"]]
        label = z["label"]
        if probs is not None:
            p = probs.get(z["name"], 0.0)
            colour = plt.cm.Blues(0.08 + 0.62 * p / max(top, 1e-9))
            label = "{} {:.0f}%".format(z["label"], 100 * p)
        ax.add_patch(Rectangle((x0, y0), z["size_x"], z["size_y"], fc=colour, ec="#777", lw=0.8, ls=":", zorder=2))
        ax.text(x0 + 0.12, y0 + z["size_y"] - 0.12, label, ha="left", va="top", fontsize=5.5 if small else 9,
                fontweight="bold", color="#222", zorder=7)
    for f in flat["furniture"]:
        ax.add_patch(Rectangle((f["x"] - f["size_x"] / 2, f["y"] - f["size_y"] / 2), f["size_x"], f["size_y"],
                               fc=FURNITURE_COLOUR, ec="#4a4036", lw=0.8, alpha=0.9, zorder=3))
        if not small:
            ax.text(f["x"], f["y"], f["name"], ha="center", va="center", fontsize=5.5, color="white", zorder=4)
    door = flat["door"]
    horizontal = door["wall"] in ("ymin", "ymax")
    half = door["width"] / 2
    xs = (door["x"] - half, door["x"] + half) if horizontal else (door["x"], door["x"])
    ys = (door["y"], door["y"]) if horizontal else (door["y"] - half, door["y"] + half)
    ax.plot(xs, ys, color="#b5651d", lw=4, solid_capstyle="butt", zorder=5)
    if order is None:
        for z in flat["zones"]:
            for spot in z["spots"]:
                a = math.radians(spot["heading_deg"])
                ax.annotate("", xy=(spot["view_x"] + 0.3 * math.cos(a), spot["view_y"] + 0.3 * math.sin(a)),
                            xytext=(spot["view_x"], spot["view_y"]),
                            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.0), zorder=6)
                ax.plot(spot["person_x"], spot["person_y"], "o", ms=2 if small else 3.5, color="#444", zorder=6)
    else:
        zones = {z["name"]: z for z in flat["zones"]}
        px = [flat["dock"]["x"]] + [zones[n]["view_x"] for n in order]
        py = [flat["dock"]["y"]] + [zones[n]["view_y"] for n in order]
        ax.plot(px, py, "-", color="#c0392b", lw=1.6, zorder=6)
        for i, n in enumerate(order):
            ax.plot(px[i + 1], py[i + 1], "o", ms=13, mfc="white", mec="#c0392b", mew=1.6, zorder=7)
            ax.text(px[i + 1], py[i + 1], str(i + 1), ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#c0392b", zorder=8)
    if truth is not None and truth != "out":
        z = next(z for z in flat["zones"] if z["name"] == truth)
        ax.plot(z["person_x"], z["person_y"], "*", ms=17, mfc="#f1c40f", mec="#333", mew=0.8, zorder=9)
    ax.plot(flat["dock"]["x"], flat["dock"]["y"], "s", ms=4 if small else 7, color="#2c7a3f", zorder=8)
    ax.set_xlim(-w / 2 - 0.15, w / 2 + 0.15)
    ax.set_ylim(-d / 2 - 0.15, d / 2 + 0.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title("flat {} · {:.1f} × {:.1f} m ({:.0f} m²)".format(flat["flat_id"], w, d, flat["area_m2"]),
                     fontsize=6 if small else 9)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--all", action="store_true", help="draw every flat, not only the first 24")
    parser.add_argument("--data", default=str(DATA))
    args = parser.parse_args()
    data = Path(args.data)
    flats = json.loads((data / "flats.json").read_text())
    out = data / "maps"
    out.mkdir(parents=True, exist_ok=True)
    chosen = flats if args.all else flats[:24]
    for flat in chosen:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        draw(ax, flat)
        fig.text(0.5, 0.01, "red arrow = robot viewing pose · dot = where she is · green square = dock · "
                 "brown = door · dark boxes = the arena's furniture", ha="center", fontsize=6, color="#666")
        fig.savefig(out / "flat_{:04d}.png".format(flat["flat_id"]), dpi=110, bbox_inches="tight")
        plt.close(fig)
    sheet = flats[:24]
    cols = 6
    rows = math.ceil(len(sheet) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.1 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, flat in zip(axes.ravel(), sheet):
        ax.axis("on")
        draw(ax, flat, small=True)
    fig.tight_layout()
    fig.savefig(out / "sheet.png", dpi=100)
    plt.close(fig)
    print("wrote {} flat plans and {}".format(len(chosen), out / "sheet.png"))


if __name__ == "__main__":
    main()
