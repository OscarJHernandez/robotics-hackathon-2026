"""Random one-room flats (the maps) for the Knock data: 3 to 5 named zones each.

A flat is a plain dict (JSON-friendly). Coordinates are the simulator's world frame: the room is centred on the
origin, x along the width, y along the depth, headings in degrees with 0 = +x, counter-clockwise (as in
app/knock.py). Sizes are at sim scale, like sim/room_furniture.xml.

Zones use the app's six names (KnockZone in app/web/src/lib/api.ts, ZONES in app/knock.py). Every flat has bed,
kitchen and bathroom; 0 to 2 of couch, armchair and entrance are added at random. Furniture stands against the
walls and every viewing pose sits in the open aisle, so the nav/ walker (no obstacle avoidance) can walk in
straight lines between poses.

Needs only numpy.
"""

import math
from collections import OrderedDict

import numpy as np

ZONE_NAMES = ("bed", "bathroom", "kitchen", "couch", "armchair", "entrance")  # app order
CLASSES = ZONE_NAMES + ("out",)  # TabPFN targets: 6 zones + not home
REQUIRED = ("bed", "kitchen", "bathroom")
OPTIONAL = ("couch", "armchair", "entrance")
OPTIONAL_WEIGHTS = (0.45, 0.35, 0.20)
N_OPTIONAL_P = (0.25, 0.40, 0.35)  # 0, 1 or 2 optional zones -> 3, 4 or 5 zones

LABELS = {"bed": "Bed", "bathroom": "Bathroom", "kitchen": "Kitchen", "couch": "Couch", "armchair": "Armchair",
          "entrance": "Entrance"}
DESCRIPTIONS = {"bed": "by the bed", "bathroom": "on the bathroom floor", "kitchen": "by the kitchen table",
                "couch": "in front of the couch", "armchair": "by the armchair", "entrance": "by the front door"}
# Solid furniture blocks the robot; the bathroom (its door / wet area) and the entrance are marked floor areas.
SOLID = {"bed", "kitchen", "couch", "armchair"}
# (length along the wall, depth into the room) in metres.
FOOTPRINT = {"bed": (0.9, 0.8), "kitchen": (0.7, 0.5), "bathroom": (0.6, 0.45), "couch": (0.8, 0.4),
             "armchair": (0.5, 0.5), "entrance": (0.6, 0.45)}

WIDTH_RANGE = (2.6, 4.5)
DEPTH_RANGE = (2.0, 3.5)
DOOR_WIDTH = 0.5
VIEW_GAP = 0.30  # viewing pose: this far in front of the zone's front edge
CLEARANCE = 0.18  # robot body: free space around a pose or the dock
PATH_MARGIN = 0.12  # straight walks between poses keep this far from solid furniture
GAP = 0.05  # between two pieces of furniture
WALLS = ("xmin", "xmax", "ymin", "ymax")
HEADING = {"xmin": 180.0, "xmax": 0.0, "ymin": -90.0, "ymax": 90.0}  # facing the wall a zone stands against
NORMAL = {"xmin": (1.0, 0.0), "xmax": (-1.0, 0.0), "ymin": (0.0, 1.0), "ymax": (0.0, -1.0)}  # into the room


# --- geometry ---

def _wall_length(wall, width, depth):
    return depth if wall in ("xmin", "xmax") else width


def _rect_on_wall(wall, centre, length, dep, width, depth):
    """Axis-aligned rectangle (x0, y0, x1, y1) of a piece standing against `wall`, centred at `centre` along it."""
    hw, hd = width / 2, depth / 2
    a0, a1 = centre - length / 2, centre + length / 2
    if wall == "ymax":
        return (a0, hd - dep, a1, hd)
    if wall == "ymin":
        return (a0, -hd, a1, -hd + dep)
    if wall == "xmax":
        return (hw - dep, a0, hw, a1)
    return (-hw, a0, -hw + dep, a1)


def _front_mid(rect, wall):
    x0, y0, x1, y1 = rect
    return {"ymax": ((x0 + x1) / 2, y0), "ymin": ((x0 + x1) / 2, y1),
            "xmax": (x0, (y0 + y1) / 2), "xmin": (x1, (y0 + y1) / 2)}[wall]


def _overlap(a, b, gap=0.0):
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def _point_rect_dist(px, py, r):
    dx = max(r[0] - px, 0.0, px - r[2])
    dy = max(r[1] - py, 0.0, py - r[3])
    return math.hypot(dx, dy)


def _inside(px, py, r, margin=0.0):
    return r[0] - margin <= px <= r[2] + margin and r[1] - margin <= py <= r[3] + margin


def _segment_clear(p, q, rects, margin):
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 0.02))
    for t in np.linspace(0.0, 1.0, n):
        x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
        if any(_inside(x, y, r, margin) for r in rects):
            return False
    return True


# --- the generator ---

def _try_layout(rng, width, depth, names):
    hw, hd = width / 2, depth / 2
    door_wall = WALLS[rng.integers(4)]
    door_len = _wall_length(door_wall, width, depth)
    door_c = float(rng.uniform(-door_len / 2 + 0.45, door_len / 2 - 0.45))
    door_rect = _rect_on_wall(door_wall, door_c, DOOR_WIDTH + 0.1, 0.6, width, depth)  # keep the doorway free

    rects = {}
    walls = {}
    order = sorted(names, key=lambda n: (n != "entrance", -FOOTPRINT[n][0] * FOOTPRINT[n][1]))
    for name in order:
        length, dep = FOOTPRINT[name]
        scale = float(rng.uniform(0.9, 1.15))
        length, dep = length * scale, dep * min(scale, 1.05)
        placed = False
        for _ in range(60):
            if name == "entrance":
                wall, centre = door_wall, door_c
            else:
                lengths = np.array([_wall_length(w, width, depth) for w in WALLS])
                wall = WALLS[rng.choice(4, p=lengths / lengths.sum())]
                wl = _wall_length(wall, width, depth)
                if wl < length + 0.1:
                    continue
                centre = float(rng.uniform(-wl / 2 + length / 2, wl / 2 - length / 2))
            r = _rect_on_wall(wall, centre, length, dep, width, depth)
            if name != "entrance" and _overlap(r, door_rect):
                continue
            if any(_overlap(r, o, GAP) for o in rects.values()):
                continue
            rects[name], walls[name] = r, wall
            placed = True
            break
        if not placed:
            return None

    # Viewing poses in the aisle, with room for the robot's body.
    solid = [rects[n] for n in rects if n in SOLID]
    poses = {}
    for name, r in rects.items():
        fx, fy = _front_mid(r, walls[name])
        nx, ny = NORMAL[walls[name]]
        vx, vy = fx + nx * VIEW_GAP, fy + ny * VIEW_GAP
        if not (-hw + CLEARANCE <= vx <= hw - CLEARANCE and -hd + CLEARANCE <= vy <= hd - CLEARANCE):
            return None
        if any(_point_rect_dist(vx, vy, s) < CLEARANCE for s in solid):
            return None
        if any(_inside(vx, vy, o) for n2, o in rects.items() if n2 != name):
            return None
        poses[name] = (vx, vy)

    nx, ny = NORMAL[door_wall]
    dx, dy = _front_mid(_rect_on_wall(door_wall, door_c, DOOR_WIDTH, 0.0, width, depth), door_wall)
    dock = (dx + nx * 0.35, dy + ny * 0.35)
    if any(_point_rect_dist(dock[0], dock[1], s) < CLEARANCE for s in solid):
        return None

    points = list(poses.values()) + [dock]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if not _segment_clear(points[i], points[j], solid, PATH_MARGIN):
                return None
    door = {"wall": door_wall, "x": round(dx, 3), "y": round(dy, 3), "width": DOOR_WIDTH}
    return rects, walls, poses, dock, door


def random_flat(rng, flat_id):
    """One random flat with 3-5 zones. `rng` is a numpy Generator; the same state always gives the same flat."""
    k = int(rng.choice(3, p=N_OPTIONAL_P))
    optional = list(rng.choice(OPTIONAL, size=k, replace=False, p=OPTIONAL_WEIGHTS)) if k else []
    names = [n for n in ZONE_NAMES if n in REQUIRED or n in optional]
    for attempt in range(400):
        if attempt % 40 == 0:
            width = float(rng.uniform(*WIDTH_RANGE))
            depth = float(rng.uniform(*DEPTH_RANGE))
        layout = _try_layout(rng, width, depth, names)
        if layout is not None:
            return _build(flat_id, width, depth, names, *layout)
    raise RuntimeError("could not lay out flat {} with zones {}".format(flat_id, names))


def _build(flat_id, width, depth, names, rects, walls, poses, dock, door):
    zones = []
    for i, name in enumerate(n for n in ZONE_NAMES if n in names):
        x0, y0, x1, y1 = rects[name]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        nx, ny = NORMAL[walls[name]]
        dep = (x1 - x0) if walls[name] in ("xmin", "xmax") else (y1 - y0)
        # Where she sits or lies: inside the zone, on the half nearest the aisle (the 15 cm camera sees her).
        px, py = cx + nx * dep / 4, cy + ny * dep / 4
        zones.append({
            "zone_id": "z{}".format(i + 1), "name": name, "label": LABELS[name], "description": DESCRIPTIONS[name],
            "solid": name in SOLID, "wall": walls[name],
            "x": round(cx, 3), "y": round(cy, 3), "size_x": round(x1 - x0, 3), "size_y": round(y1 - y0, 3),
            "view_x": round(poses[name][0], 3), "view_y": round(poses[name][1], 3),
            "heading_deg": HEADING[walls[name]],
            "person_x": round(px, 3), "person_y": round(py, 3),
        })
    k = 1 if len(zones) == 3 else 2
    for z in zones:
        others = sorted((math.hypot(o["x"] - z["x"], o["y"] - z["y"]), o["name"]) for o in zones if o is not z)
        z["neighbours"] = [n for _, n in others[:k]]
    return {"flat_id": flat_id, "width": round(width, 3), "depth": round(depth, 3),
            "area_m2": round(width * depth, 3), "door": door,
            "dock": {"x": round(dock[0], 3), "y": round(dock[1], 3)}, "zones": zones}


# --- helpers used by the generator, the prior and the app follow-up ---

def zone_names(flat):
    return [z["name"] for z in flat["zones"]]


def zone(flat, name):
    for z in flat["zones"]:
        if z["name"] == name:
            return z
    raise KeyError(name)


def signature(flat):
    """The flat's zone set as a string, e.g. 'bed,bathroom,kitchen,couch' (app order)."""
    return ",".join(zone_names(flat))


def view_xy(flat, name):
    if name == "dock":
        return flat["dock"]["x"], flat["dock"]["y"]
    z = zone(flat, name)
    return z["view_x"], z["view_y"]


def distance(flat, a, b):
    """Metres between two viewing poses (or 'dock'): the walker goes in a straight line in the aisle."""
    (ax, ay), (bx, by) = view_xy(flat, a), view_xy(flat, b)
    return math.hypot(ax - bx, ay - by)


def bearing_from_dock(flat, x, y):
    """Degrees from the dock to (x, y), 0 = +x, counter-clockwise, in [0, 360)."""
    return math.degrees(math.atan2(y - flat["dock"]["y"], x - flat["dock"]["x"])) % 360.0


def nearest_to_door(flat, among=None):
    names = among or zone_names(flat)
    dx, dy = flat["door"]["x"], flat["door"]["y"]
    return min(names, key=lambda n: math.hypot(zone(flat, n)["x"] - dx, zone(flat, n)["y"] - dy))


def to_app_zones(flat):
    """The flat in the shape of app/knock.py's ZONES: name -> (label, (x, y), heading_deg, description)."""
    return OrderedDict((z["name"], (z["label"], (z["view_x"], z["view_y"]), z["heading_deg"], z["description"]))
                       for z in flat["zones"])
