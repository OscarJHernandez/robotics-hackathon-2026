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
COLOURS = {"bed": "#8fb0e0", "bathroom": "#9fd8d0", "kitchen": "#e0c08a", "couch": "#e29a8e",
           "armchair": "#9ccb9c", "entrance": "#d8d8a0"}


def draw(ax, flat, title=True, small=False):
    w, d = flat["width"], flat["depth"]
    ax.add_patch(Rectangle((-w / 2, -d / 2), w, d, fill=True, fc="#f6f3ee", ec="#444", lw=2))
    door = flat["door"]
    horizontal = door["wall"] in ("ymin", "ymax")
    half = door["width"] / 2
    xs = (door["x"] - half, door["x"] + half) if horizontal else (door["x"], door["x"])
    ys = (door["y"], door["y"]) if horizontal else (door["y"] - half, door["y"] + half)
    ax.plot(xs, ys, color="#f6f3ee", lw=5, solid_capstyle="butt", zorder=3)
    ax.plot(xs, ys, color="#b5651d", lw=1.5, ls="--", zorder=4)
    for z in flat["zones"]:
        x0, y0 = z["x"] - z["size_x"] / 2, z["y"] - z["size_y"] / 2
        ax.add_patch(Rectangle((x0, y0), z["size_x"], z["size_y"], fc=COLOURS[z["name"]], ec="#555",
                               lw=1, ls="-" if z["solid"] else ":", alpha=0.95, zorder=2))
        ax.text(z["x"], z["y"], z["label"], ha="center", va="center", fontsize=5.5 if small else 8, zorder=5)
        a = math.radians(z["heading_deg"])
        ax.annotate("", xy=(z["view_x"] + 0.16 * math.cos(a), z["view_y"] + 0.16 * math.sin(a)),
                    xytext=(z["view_x"], z["view_y"]),
                    arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.2), zorder=6)
        ax.plot(z["person_x"], z["person_y"], "o", ms=2.5 if small else 4, color="#444", zorder=6)
    ax.plot(flat["dock"]["x"], flat["dock"]["y"], "s", ms=4 if small else 6, color="#2c7a3f", zorder=6)
    ax.set_xlim(-w / 2 - 0.15, w / 2 + 0.15)
    ax.set_ylim(-d / 2 - 0.15, d / 2 + 0.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        names = ", ".join(z["name"] for z in flat["zones"])
        ax.set_title("flat {} · {:.1f} × {:.1f} m · {} zones".format(flat["flat_id"], w, d, len(flat["zones"])),
                     fontsize=6 if small else 9)
        if not small:
            ax.set_xlabel(names, fontsize=7)


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
        fig, ax = plt.subplots(figsize=(5, 4))
        draw(ax, flat)
        fig.text(0.5, 0.01, "red arrow = robot viewing pose · dot = where she sits · green square = dock · "
                 "dashed brown = door", ha="center", fontsize=6, color="#666")
        fig.savefig(out / "flat_{:04d}.png".format(flat["flat_id"]), dpi=110, bbox_inches="tight")
        plt.close(fig)
    sheet = flats[:24]
    cols = 6
    rows = math.ceil(len(sheet) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
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
