"""One worked example of the Bayesian search as a picture: the inputs, the belief step by step, the search on the
floor plan, and the outcome. Used by search_demo.py.

Layout (16 x 9): a header with the story in one sentence; left, the three inputs (stated habits, 4 weeks of journal,
last heard); middle, the belief after each step as stacked bars; right, the floor plan with the search path; bottom
left, how far the robot walked against two simple rules.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

import search as S  # noqa: E402
from flats import CLASSES, LABELS  # noqa: E402
from residents import SLOTS  # noqa: E402

# Colors: categorical slots 1-5 of the reference palette (validated: adjacent CVD dE >= 9.1, normal >= 19.6; three
# sit below 3:1 on the surface, so every colored mark here carries a text label). `out` is a neutral, not a series.
ZONE_COLOR = {"bedroom": "#2a78d6", "living_room": "#eb6834", "dining_area": "#1baf7a", "kitchen": "#eda100",
              "hallway": "#e87ba4", "out": "#bdbbb3"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
FURNITURE = "#d9d7cf"
HERE = "#e34948"  # where she really was
NAMES = dict(LABELS, out="Out")
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
STAGE_LABELS = ("Population", "+ stated habits", "+ journal", "+ last heard")


def _font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Avenir Next", "Helvetica Neue", "Arial"):
        if name in available:
            return name
    return "DejaVu Sans"


FONT = _font()
plt.rcParams.update({"font.family": FONT, "text.color": INK, "axes.edgecolor": HAIRLINE,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def tint(hex_color, amount):
    """Mix a color with the surface: amount 0 = surface, 1 = the color."""
    c = matplotlib.colors.to_rgb(hex_color)
    s = matplotlib.colors.to_rgb(SURFACE)
    return tuple(s[i] + (c[i] - s[i]) * amount for i in range(3))


def clock(hour):
    return "{:02d}:{:02d}".format(int(hour), int(round((hour % 1) * 60)) % 60)


def ago(minutes):
    h, m = divmod(int(round(minutes)), 60)
    return "{} h {:02d} min".format(h, m) if h else "{} min".format(m)


def _card(ax, title, subtitle=None):
    ax.set_axis_off()
    ax.text(0.0, 1.0, title.upper(), transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK_2,
            va="top", ha="left")
    if subtitle:
        ax.text(1.0, 1.0, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, va="top", ha="right")


def _box(ax, x, y, w, h, fc, ec="none", lw=0.0, r=0.012, z=2, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size={}".format(r), fc=fc, ec=ec,
                                lw=lw, zorder=z, transform=ax.transAxes, **kw))


# --- the inputs ---

def habits_card(ax, r):
    wk_now = "1" if r["weekday"] >= 5 else "0"
    _card(ax, "Stated habits", "from a form; can be wrong")
    x0, y_top, row_h, col_w, gap = 0.20, 0.80, 0.24, 0.265, 0.012
    for j, slot in enumerate(SLOTS):
        ax.text(x0 + j * col_w + col_w / 2, y_top + 0.04, slot.capitalize(), transform=ax.transAxes, fontsize=9,
                color=MUTED, ha="center", va="bottom")
    for i, (wk, label) in enumerate((("0", "Weekdays"), ("1", "Weekends"))):
        y = y_top - (i + 1) * row_h
        ax.text(0.0, y + row_h / 2 - gap, label, transform=ax.transAxes, fontsize=10, color=INK_2, va="center")
        for j, zone in enumerate(r["habits"]["usual"][wk]):
            now = wk == wk_now and j == r["slot"]
            _box(ax, x0 + j * col_w + gap, y, col_w - 2 * gap, row_h - 2 * gap, tint(ZONE_COLOR[zone], 0.28),
                 ec=INK if now else "none", lw=1.8 if now else 0)
            ax.text(x0 + j * col_w + col_w / 2, y + row_h / 2 - gap, NAMES[zone], transform=ax.transAxes,
                    fontsize=10, ha="center", va="center", fontweight="bold" if now else "normal", zorder=3)
    outings = r["habits"]["outings"]
    text = ", ".join("{} {}".format(DAYS[wd], SLOTS[s]) for wd, s in outings) if outings else "none"
    ax.text(0.0, 0.0, "Regular outings: " + text, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom",
            wrap=True)


def journal_card(ax, r):
    d = r["diary"].sort_values(["day", "slot"])
    days = sorted(d["day"].unique())
    weekend_now = r["weekday"] >= 5
    _card(ax, "Journal", "{} days, where the robot heard {}".format(len(days), r["persona"]["name"]))
    x0, x1, y_top = 0.20, 1.0, 0.84
    cw = (x1 - x0) / len(days)
    ch = 0.13
    cells = {(int(a), int(b)): z for a, b, z in zip(d["day"], d["slot"], d["zone"])}
    weekday_of = dict(zip(d["day"].astype(int), d["weekday"].astype(int)))
    for i, slot in enumerate(SLOTS):
        y = y_top - (i + 1) * (ch + 0.02)
        now = i == r["slot"]
        ax.text(0.0, y + ch / 2, slot.capitalize(), transform=ax.transAxes, fontsize=9.5, va="center",
                color=INK if now else INK_2, fontweight="bold" if now else "normal")
        for k, day in enumerate(days):
            zone = cells.get((int(day), i))
            if zone is None:
                continue
            counts = now and (weekday_of[int(day)] >= 5) == weekend_now
            color = ZONE_COLOR[zone]
            _box(ax, x0 + k * cw + cw * 0.08, y, cw * 0.84, ch, color if counts else tint(color, 0.30), r=0.004)
    for k, day in enumerate(days):  # day initials under the strip, weekends darker
        wd = weekday_of[int(day)]
        ax.text(x0 + k * cw + cw / 2, y_top - 3 * (ch + 0.02) - 0.035, DAYS[wd][0], transform=ax.transAxes,
                fontsize=6.5, ha="center", va="top", color=INK_2 if wd >= 5 else MUTED)
    same = d[(d["slot"] == r["slot"]) & ((d["weekday"] >= 5) == weekend_now)]
    counts = same["zone"].value_counts()
    kind = "weekend" if weekend_now else "weekday"
    ax.text(0.0, 0.0, "Full color: the {} {} {}s that count\n".format(len(same), kind, SLOTS[r["slot"]])
            + " · ".join("{} {}".format(NAMES[z], n) for z, n in counts.items()),
            transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom", linespacing=1.5)


def last_heard_card(ax, r):
    _card(ax, "Last heard")
    if not r["last"]:
        ax.text(0.0, 0.25, "Not heard yet", transform=ax.transAxes, fontsize=11, color=INK_2)
        return
    zone, minutes = r["last"]
    ax.plot([0.018], [0.36], "o", ms=12, color=ZONE_COLOR[zone], transform=ax.transAxes, clip_on=False)
    ax.text(0.06, 0.36, "{}, {} ago".format(NAMES[zone], ago(minutes)), transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="center")
    note = ("this {}: {} has most likely not moved".format(SLOTS[r["slot"]], r["persona"]["name"]) if r["recent"]
            else "an earlier part of the day: says little about now")
    ax.text(0.06, 0.02, note, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


# --- the belief ---

def belief_card(ax, r):
    ax.set_axis_off()
    ax.text(0.0, 1.0, "HOW THE BELIEF BUILDS UP", transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK_2, va="top")
    # legend: one row of swatches, in class order
    slots = (0.0, 0.33, 0.66)  # two rows of three
    for i, c in enumerate(CLASSES):
        x, y = slots[i % 3], 0.925 - 0.045 * (i // 3)
        ax.add_patch(Rectangle((x, y - 0.011), 0.03, 0.022, transform=ax.transAxes, fc=ZONE_COLOR[c], ec="none"))
        ax.text(x + 0.045, y, NAMES[c], transform=ax.transAxes, fontsize=9, va="center", color=INK_2)
    bar_h, top = 0.095, 0.785
    for i, (stage, label) in enumerate(zip(S.STAGES, STAGE_LABELS)):
        y = top - i * 0.175 - bar_h
        ax.text(0.0, y + bar_h + 0.012, label, transform=ax.transAxes, fontsize=10, va="bottom",
                fontweight="bold" if i == 3 else "normal", color=INK if i == 3 else INK_2)
        probs = r["beliefs"][stage]
        x = 0.0
        for c in CLASSES:
            w = probs[c]
            if w <= 0.0005:
                continue
            ax.add_patch(Rectangle((x, y), w, bar_h, transform=ax.transAxes, fc=ZONE_COLOR[c], ec=SURFACE, lw=1.6))
            if w >= 0.075:
                ax.text(x + w / 2, y + bar_h / 2, "{:.0f}%".format(100 * w), transform=ax.transAxes, fontsize=9,
                        ha="center", va="center", color=INK)
            x += w
    ax.text(0.0, 0.0, change_note(r), transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom", wrap=True)


def change_note(r):
    """The step that changed the belief most, and the zone that moved most in it."""
    b = r["beliefs"]
    best = None
    for prev, stage, label in zip(S.STAGES, S.STAGES[1:], STAGE_LABELS[1:]):
        for c in CLASSES:
            delta = b[stage][c] - b[prev][c]
            if best is None or abs(delta) > abs(best[0]):
                best = (delta, label, c, b[prev][c], b[stage][c])
    delta, label, c, a, z = best
    if abs(delta) < 0.03:
        return "No step changes the belief much."
    return "Biggest change: {} moves {} from {:.0f}% to {:.0f}%.".format(label.lstrip("+ "), NAMES[c], 100 * a, 100 * z)


# --- the floor plan ---

def plan_card(ax, r, order):
    flat = r["flat"]
    probs = r["beliefs"]["last_heard"]
    h = flat["width"] / 2
    ax.set_xlim(-h - 0.05, h + 0.05)
    ax.set_ylim(-h - 0.05, h + 0.05)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for z in flat["zones"]:
        x0, y0 = z["x"] - z["size_x"] / 2, z["y"] - z["size_y"] / 2
        ax.add_patch(Rectangle((x0, y0), z["size_x"], z["size_y"], fc=tint(ZONE_COLOR[z["name"]], 0.16),
                               ec=SURFACE, lw=3, zorder=1))
        tx, ty = x0 + 0.14, y0 + z["size_y"] - 0.12
        box = dict(boxstyle="round,pad=0.15", fc=tint(ZONE_COLOR[z["name"]], 0.16), ec="none")
        sep = "\n" if z["size_x"] < 1.9 else "  "  # a narrow zone gets two lines
        ax.text(tx, ty, "{}{}{:.0f}%".format(NAMES[z["name"]], sep, 100 * probs[z["name"]]), fontsize=10.5,
                fontweight="bold", va="top", zorder=14, bbox=box)
    for f in flat["furniture"]:
        ax.add_patch(FancyBboxPatch((f["x"] - f["size_x"] / 2, f["y"] - f["size_y"] / 2), f["size_x"], f["size_y"],
                                    boxstyle="round,pad=0,rounding_size=0.06", fc=FURNITURE, ec="none", zorder=3))
        if f["name"] in ("bed", "couch", "table", "cabinet", "armchair"):
            ax.text(f["x"], f["y"], f["name"], fontsize=7, color=MUTED, ha="center", va="center", zorder=4)
    ax.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fc="none", ec=INK_2, lw=2.5, zorder=5))
    door = flat["door"]
    half = door["width"] / 2
    if door["wall"] in ("ymin", "ymax"):
        xs, ys = (door["x"] - half, door["x"] + half), (door["y"], door["y"])
    else:
        xs, ys = (door["x"], door["x"]), (door["y"] - half, door["y"] + half)
    ax.plot(xs, ys, color=SURFACE, lw=5, solid_capstyle="butt", zorder=6)
    ax.plot(xs, ys, color=MUTED, lw=1.2, ls=(0, (2, 2)), zorder=6)

    zones = {z["name"]: z for z in flat["zones"]}
    dock = (flat["dock"]["x"], flat["dock"]["y"])
    if r["truth"] == "out":  # no path: the point is that the robot should not need to walk
        order = []
        ax.text(0, -h + 0.45, "Nobody home. {:.0f}% not home before the robot starts;\nwalking every zone would "
                "take {:.1f} m.".format(100 * probs["out"], r["walks"]["full"][0]), fontsize=10.5, ha="center",
                va="bottom", zorder=15, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.8,rounding_size=0.3", fc=SURFACE, ec=HAIRLINE, lw=1))
    px = [dock[0]] + [zones[n]["view_x"] for n in order]
    py = [dock[1]] + [zones[n]["view_y"] for n in order]
    ax.plot(px, py, "-", color=INK, lw=2, zorder=9, solid_capstyle="round")
    ax.add_patch(FancyBboxPatch((dock[0] - 0.13, dock[1] - 0.13), 0.26, 0.26, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=INK, ec=SURFACE, lw=2, zorder=10))
    side = 1 if dock[0] < 0 else -1
    ax.text(dock[0] + side * 0.24, dock[1], "dock", fontsize=8.5, color=INK_2, ha="left" if side > 0 else "right",
            va="center", zorder=10)
    now = dict(probs)  # the belief when the robot picks each stop (it changes after every miss)
    for i, n in enumerate(order):
        ax.add_patch(Circle((px[i + 1], py[i + 1]), 0.2, fc=SURFACE, ec=INK, lw=2, zorder=11))
        ax.text(px[i + 1], py[i + 1], str(i + 1), fontsize=10, fontweight="bold", ha="center", va="center", zorder=12)
        right = px[i + 1] < h - 1.0
        ax.text(px[i + 1] + (0.28 if right else -0.28), py[i + 1], "{:.0f}%".format(100 * now[n]), fontsize=9,
                ha="left" if right else "right", va="center", color=INK, zorder=12,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
        now = S.after_miss(now, n)
    if r["truth"] != "out":
        x, y = r["here"]
        ax.plot(x, y, marker="*", ms=20, mfc=HERE, mec=SURFACE, mew=1.5, zorder=13)
        ax.text(x, y - 0.3, "{} was here".format(r["persona"]["name"]), fontsize=8.5, color=INK, ha="center",
                va="top", zorder=13, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))


# --- the outcome ---

def tiles(fig, rect, r):
    x0, y0, w, h = rect
    found = r["truth"] != "out"
    metres = r["walks"]["full"][0]
    def value(key):
        n = r["walks"][key][1]
        return "{} zone{}".format(n, "" if n == 1 else "s") if found else "all 5 zones"

    items = [("This search: most likely first", value("full"),
              "{:.1f} m walked".format(metres) if found else "nobody home: whole round, {:.1f} m".format(metres)),
             ("Rule: nearest zone first", value("nearest"), "{:.1f} m walked".format(r["walks"]["nearest"][0])),
             ("Rule: last heard first", value("last"), "{:.1f} m walked".format(r["walks"]["last"][0]))]
    tw = (w - 0.02) / 3
    for i, (label, value, sub) in enumerate(items):
        ax = fig.add_axes((x0 + i * (tw + 0.01), y0, tw, h))
        ax.set_axis_off()
        _box(ax, 0, 0, 1, 1, "#f1f0ec" if i else tint("#2a78d6", 0.12), r=0.06, z=1)
        ax.text(0.07, 0.78, label, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="center")
        ax.text(0.07, 0.42, value, transform=ax.transAxes, fontsize=20 if i == 0 else 16,
                fontweight="bold" if i == 0 else "normal", va="center", color=INK if i == 0 else INK_2)
        ax.text(0.07, 0.14, sub, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="center")


def figure(title, story, r, path):
    order = r["orders"]["full"]
    if r["truth"] in order:
        order = order[:order.index(r["truth"]) + 1]  # the robot stops when it finds her
    fig = plt.figure(figsize=(16, 9), dpi=100)
    name = r["persona"]["name"]
    when = "{} {} · {}".format(DAYS[r["weekday"]], clock(r["hour"]), SLOTS[r["slot"]])
    fig.text(0.035, 0.945, title, fontsize=21, fontweight="bold", va="center")
    fig.text(0.965, 0.945, "{} · {}".format(name, when), fontsize=13, color=INK_2, va="center", ha="right")
    fig.text(0.035, 0.892, story, fontsize=12.5, color=INK_2, va="center")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.858, 0.858], color=HAIRLINE, lw=1))

    habits_card(fig.add_axes((0.035, 0.605, 0.30, 0.215)), r)
    journal_card(fig.add_axes((0.035, 0.345, 0.30, 0.225)), r)
    last_heard_card(fig.add_axes((0.035, 0.225, 0.30, 0.085)), r)
    belief_card(fig.add_axes((0.375, 0.225, 0.255, 0.595)), r)
    tiles(fig, (0.035, 0.045, 0.595, 0.13), r)
    ax = fig.add_axes((0.655, 0.2, 0.31, 0.59))
    ax.set_anchor("N")
    plan_card(ax, r, order)
    fig.text(0.655, 0.81, "THE SEARCH: MOST LIKELY ZONE FIRST", fontsize=9, fontweight="bold",
             color=INK_2, ha="left", va="center")
    key = fig.add_axes((0.655, 0.045, 0.31, 0.13))
    key.set_axis_off()
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    rows = (("stop", "Stop 1, 2, ... in order; the % is its chance when the robot picks it"),
            ("star", "Where {} really was".format(name) if r["truth"] != "out" else "{} was out".format(name)),
            ("dock", "The robot's dock, where every search starts"))
    for i, (kind, text) in enumerate(rows):
        y = 0.78 - i * 0.3
        if kind == "stop":
            key.plot([0.025], [y], "o", ms=13, mfc=SURFACE, mec=INK, mew=2)
            key.text(0.025, y, "1", fontsize=8, fontweight="bold", ha="center", va="center")
        elif kind == "star":
            if r["truth"] != "out":
                key.plot([0.025], [y], "*", ms=15, mfc=HERE, mec=SURFACE)
        else:
            key.plot([0.025], [y], "s", ms=9, color=INK)
        key.text(0.065, y, text, fontsize=9.5, color=INK_2, va="center")
    fig.savefig(path, dpi=150)
    plt.close(fig)
