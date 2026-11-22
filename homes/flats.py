"""Random studio flats (the floor plans) for the Knock data, built on the MuJoCo arena.

A flat is the simulator's arena (sim/arena_furniture.xml): one closed, square room of 50 m², centred on the origin,
with the arena's 8 pieces of furniture. The floor is divided into five named zones, and every point of the floor
belongs to exactly one of them:

    bedroom       bed
    living_room   couch, armchair
    dining_area   table, stool
    kitchen       cabinet (stands in for the kitchen counter)
    hallway       crate, plant pot, the door and the robot's dock

The hallway is a strip along one wall; the other four zones are the quarters of the rest of the room (kitchen and
dining area always share an edge). The split lines, which quarter is which, the furniture positions and which wall
the hallway is on are random. Furniture stands with its back to a wall of the room, except the table and its stool.

A flat is a plain dict (JSON-friendly). Coordinates are the simulator's world frame, in metres; headings in degrees
with 0 = +x, counter-clockwise (as in app/knock.py). `arena_layout(flat)` gives the layout `Arena.apply()` in
vision/scene.py takes, so the same flat can be opened in MuJoCo. The arena has no real door: `door` is where the
door would be on the (closed) wall.

Needs only numpy.
"""

import math
from collections import OrderedDict

ZONE_NAMES = ("bedroom", "living_room", "dining_area", "kitchen", "hallway")
CLASSES = ZONE_NAMES + ("out",)  # TabPFN targets: 5 zones + not home
LABELS = {"bedroom": "Bedroom", "living_room": "Living room", "dining_area": "Dining area", "kitchen": "Kitchen",
          "hallway": "Hallway"}
DESCRIPTIONS = {"bedroom": "by the bed", "living_room": "in front of the couch", "dining_area": "by the dining table",
                "kitchen": "by the kitchen cabinet", "hallway": "by the front door"}
ZONE_FURNITURE = {"bedroom": ("bed",), "living_room": ("couch", "armchair"), "dining_area": ("table", "stool"),
                  "kitchen": ("cabinet",), "hallway": ("crate", "plant")}

AREA = 50.0  # m², as vision/scene.py
HALF = math.sqrt(AREA) / 2  # 3.5355 m from the centre to the inside of each wall

# The arena's furniture (sim/arena_furniture.xml). body: the mocap body; half: half the footprint along the piece's
# own x and y; fronts: the directions (degrees, in the piece's own frame) that may face the room when it stands
# against a wall; radius: the circle around the footprint that vision/scene.py uses; color: the XML's color.
PIECES = {
    "bed": dict(body="obs_0", half=(0.9, 0.5), fronts=(90.0, -90.0), radius=1.3616, color=(0.55, 0.65, 0.85)),
    "couch": dict(body="obs_1", half=(0.8, 0.35), fronts=(90.0,), radius=1.0831, color=(0.75, 0.5, 0.3)),
    "cabinet": dict(body="obs_2", half=(0.4, 0.25), fronts=(90.0, -90.0), radius=0.4717, color=(0.5, 0.38, 0.25)),
    "stool": dict(body="obs_3", half=(0.18, 0.18), fronts=(0.0,), radius=0.18, color=(0.3, 0.6, 0.6)),
    "table": dict(body="obs_4", half=(0.5, 0.35), fronts=(90.0,), radius=0.6103, color=(0.7, 0.6, 0.4)),
    "armchair": dict(body="obs_5", half=(0.3, 0.3), fronts=(0.0,), radius=0.5627, color=(0.45, 0.6, 0.45)),
    "crate": dict(body="obs_6", half=(0.25, 0.25), fronts=(90.0, -90.0), radius=0.3536, color=(0.6, 0.5, 0.7)),
    "plant": dict(body="obs_7", half=(0.15, 0.15), fronts=(0.0,), radius=0.15, color=(0.35, 0.35, 0.4)),
}

HALL_DEPTH = (1.3, 1.6)  # metres: how far the hallway strip reaches into the room
SPLIT_X = 0.5  # the left/right split line is this far from the middle at most
SPLIT_Y = 0.3  # the same for the split between the two rows of quarters
DOOR_WIDTH = 0.9
DOOR_MARGIN = 1.2  # the door's centre stays this far from the room's corners
DOCK_SIDE = 0.9  # the dock is this far beside the door, along the wall
DOCK_DEPTH = 0.4  # the dock and the spot by the door are this far from the wall
WALL_GAP = 0.02  # between a piece of furniture and the wall it stands against
EDGE = 0.10  # furniture keeps this far from the edge of its zone
FURNITURE_GAP = 0.10  # between two pieces of furniture
PERSON_GAP = 0.30  # where she is: this far in front of the furniture, on the floor
VIEW_GAP = 0.90  # the robot's viewing pose: this far in front of the furniture, looking at her
CLEARANCE = 0.30  # robot body (ROBOT_RADIUS in vision/scene.py): free space around a viewing pose and the dock
TABLE_MARGIN = 0.9  # the table's centre keeps this far from the edge of the dining area
WALLS = {"xmin": ((1.0, 0.0), 0.0), "xmax": ((-1.0, 0.0), 180.0), "ymin": ((0.0, 1.0), 90.0),
         "ymax": ((0.0, -1.0), -90.0)}  # wall: (direction into the room, that direction in degrees)
DOOR_WALLS = ("ymin", "xmax", "ymax", "xmin")  # the hallway's wall after 0, 1, 2, 3 quarter turns


# --- geometry ---

def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _half_extents(half, yaw):
    """Half the footprint along world x and y of a piece turned by `yaw` (a multiple of 90 degrees)."""
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return c * half[0] + s * half[1], s * half[0] + c * half[1]


def _overlap(a, b, gap=0.0):
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def _point_rect_dist(px, py, r):
    dx = max(r[0] - px, 0.0, px - r[2])
    dy = max(r[1] - py, 0.0, py - r[3])
    return math.hypot(dx, dy)


def _inside(px, py, r, margin=0.0):
    return r[0] - margin <= px <= r[2] + margin and r[1] - margin <= py <= r[3] + margin


def _rect_inside(inner, outer, tol=1e-9):
    return (outer[0] - tol <= inner[0] and inner[2] <= outer[2] + tol
            and outer[1] - tol <= inner[1] and inner[3] <= outer[3] + tol)


def _turn(k, x, y):
    """(x, y) after k quarter turns counter-clockwise around the centre of the room."""
    for _ in range(k % 4):
        x, y = -y, x
    return x, y


def _turn_rect(k, r):
    (ax, ay), (bx, by) = _turn(k, r[0], r[1]), _turn(k, r[2], r[3])
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def _shared_edge(a, b):
    """Length of the border two zone rectangles share (0 when they only touch at a corner or not at all)."""
    tol = 1e-6
    if abs(a[2] - b[0]) < tol or abs(b[2] - a[0]) < tol:
        return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if abs(a[3] - b[1]) < tol or abs(b[3] - a[1]) < tol:
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    return 0.0


# --- placing furniture (all in the frame where the hallway is along the ymin wall) ---

def _outer_walls(rect):
    """The walls of the room that this zone rectangle lies against."""
    tol = 1e-6
    return [w for w, hit in (("xmin", abs(rect[0] + HALF) < tol), ("xmax", abs(rect[2] - HALF) < tol),
                             ("ymin", abs(rect[1] + HALF) < tol), ("ymax", abs(rect[3] - HALF) < tol)) if hit]


def _against_wall(rng, name, rect, wall):
    """`name` with its back to `wall`, somewhere along the part of the wall inside the zone `rect`. None: no room."""
    piece = PIECES[name]
    (nx, ny), angle = WALLS[wall]
    yaw = math.radians(angle - piece["fronts"][int(rng.integers(len(piece["fronts"])))])
    hx, hy = _half_extents(piece["half"], yaw)
    if wall in ("xmin", "xmax"):
        lo, hi = rect[1] + EDGE + hy, rect[3] - EDGE - hy
        if lo > hi:
            return None
        cx = rect[0] + WALL_GAP + hx if wall == "xmin" else rect[2] - WALL_GAP - hx
        cy, depth = float(rng.uniform(lo, hi)), hx
    else:
        lo, hi = rect[0] + EDGE + hx, rect[2] - EDGE - hx
        if lo > hi:
            return None
        cy = rect[1] + WALL_GAP + hy if wall == "ymin" else rect[3] - WALL_GAP - hy
        cx, depth = float(rng.uniform(lo, hi)), hy
    return {"name": name, "x": cx, "y": cy, "yaw": yaw, "rect": (cx - hx, cy - hy, cx + hx, cy + hy),
            "front": (cx + nx * depth, cy + ny * depth), "normal": (nx, ny)}


def _spot(name, front, normal):
    """Where she is in front of a piece of furniture, and the robot's viewing pose looking at her."""
    (fx, fy), (nx, ny) = front, normal
    return {"name": name, "person": (fx + nx * PERSON_GAP, fy + ny * PERSON_GAP),
            "view": (fx + nx * VIEW_GAP, fy + ny * VIEW_GAP), "heading": _wrap(math.degrees(math.atan2(-ny, -nx)))}


def _place(rng, name, rect, taken, tries=40):
    walls = _outer_walls(rect)
    for _ in range(tries):
        p = _against_wall(rng, name, rect, walls[int(rng.integers(len(walls)))])
        if p is not None and not any(_overlap(p["rect"], t["rect"], FURNITURE_GAP) for t in taken):
            return p
    return None


def _table_and_stool(rng, rect):
    """The table somewhere in the middle of the dining area, the stool at one long side, her spot at the other."""
    if rect[2] - rect[0] < 2 * TABLE_MARGIN or rect[3] - rect[1] < 2 * TABLE_MARGIN:
        return None
    cx = float(rng.uniform(rect[0] + TABLE_MARGIN, rect[2] - TABLE_MARGIN))
    cy = float(rng.uniform(rect[1] + TABLE_MARGIN, rect[3] - TABLE_MARGIN))
    yaw = math.radians(90.0 * int(rng.integers(2)))
    side = 1.0 if rng.random() < 0.5 else -1.0
    along = (math.cos(yaw), math.sin(yaw))  # the table's long axis
    across = (-math.sin(yaw) * side, math.cos(yaw) * side)  # towards the stool
    half = PIECES["table"]["half"]
    hx, hy = _half_extents(half, yaw)
    table = {"name": "table", "x": cx, "y": cy, "yaw": yaw, "rect": (cx - hx, cy - hy, cx + hx, cy + hy)}
    r = PIECES["stool"]["half"][0]
    reach, shift = half[1] + r + FURNITURE_GAP + 0.02, float(rng.uniform(-0.25, 0.25))
    sx, sy = cx + across[0] * reach + along[0] * shift, cy + across[1] * reach + along[1] * shift
    stool = {"name": "stool", "x": sx, "y": sy, "yaw": 0.0, "rect": (sx - r, sy - r, sx + r, sy + r)}
    front = (cx - across[0] * half[1], cy - across[1] * half[1])
    return table, stool, _spot("table", front, (-across[0], -across[1]))


def _try_layout(rng):
    hall = float(rng.uniform(*HALL_DEPTH))
    split_x = float(rng.uniform(-SPLIT_X, SPLIT_X))
    y0 = -HALF + hall
    split_y = (y0 + HALF) / 2 + float(rng.uniform(-SPLIT_Y, SPLIT_Y))
    quarters = [(-HALF, y0, split_x, split_y), (split_x, y0, HALF, split_y),  # going round: sw, se, ne, nw
                (split_x, split_y, HALF, HALF), (-HALF, split_y, split_x, HALF)]
    while True:
        order = [int(i) for i in rng.permutation(4)]
        rooms = dict(zip(("bedroom", "living_room", "dining_area", "kitchen"), order))
        if (rooms["kitchen"] - rooms["dining_area"]) % 2 == 1:  # next to each other, not across the middle
            break
    rects = {name: quarters[q] for name, q in rooms.items()}
    rects["hallway"] = (-HALF, -HALF, HALF, y0)

    furniture, spots = {}, {name: [] for name in ZONE_NAMES}
    for zone, names in (("bedroom", ("bed",)), ("living_room", ("couch", "armchair")), ("kitchen", ("cabinet",))):
        for name in names:
            p = _place(rng, name, rects[zone], [furniture[n] for n in names if n in furniture])
            if p is None:
                return None
            furniture[name] = p
            spots[zone].append(_spot(name, p["front"], p["normal"]))
    dining = _table_and_stool(rng, rects["dining_area"])
    if dining is None:
        return None
    furniture["table"], furniture["stool"], spot = dining
    spots["dining_area"].append(spot)

    door_x = float(rng.uniform(-HALF + DOOR_MARGIN, HALF - DOOR_MARGIN))
    side = 1.0 if rng.random() < 0.5 else -1.0
    dock = (door_x + side * DOCK_SIDE, -HALF + DOCK_DEPTH)
    spots["hallway"].append(_spot("door", (door_x, -HALF + DOCK_DEPTH - PERSON_GAP), (0.0, 1.0)))
    doorway = (door_x - DOOR_WIDTH / 2, -HALF, door_x + DOOR_WIDTH / 2, -HALF + 0.8)  # kept free
    for name in ("crate", "plant"):
        for _ in range(40):
            p = _place(rng, name, rects["hallway"], [furniture[n] for n in ("crate", "plant") if n in furniture])
            if p is not None and not _overlap(p["rect"], doorway, FURNITURE_GAP):
                break
        else:
            return None
        furniture[name] = p

    # Checks: furniture inside its zone and apart, her spots on free floor, room for the robot at every pose.
    solid = [f["rect"] for f in furniture.values()]
    for zone, names in ZONE_FURNITURE.items():
        if not all(_rect_inside(furniture[n]["rect"], rects[zone]) for n in names):
            return None
    names = list(furniture)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _overlap(furniture[a]["rect"], furniture[b]["rect"], FURNITURE_GAP - 1e-9):
                return None
    wall = HALF - CLEARANCE
    for zone, zone_spots in spots.items():
        for s in zone_spots:
            (px, py), (vx, vy) = s["person"], s["view"]
            if not (_inside(px, py, rects[zone]) and _inside(vx, vy, rects[zone])):
                return None
            if max(abs(vx), abs(vy)) > wall or max(abs(px), abs(py)) > HALF - 0.1:
                return None
            if any(_point_rect_dist(px, py, r) < 0.12 or _point_rect_dist(vx, vy, r) < CLEARANCE for r in solid):
                return None
    if any(_point_rect_dist(dock[0], dock[1], r) < CLEARANCE for r in solid):
        return None
    return rects, furniture, spots, (door_x, -HALF), dock


def random_flat(rng, flat_id):
    """One random flat with the five zones. `rng` is a numpy Generator; the same state always gives the same flat."""
    for _ in range(400):
        layout = _try_layout(rng)
        if layout is not None:
            return _build(flat_id, int(rng.integers(4)), *layout)
    raise RuntimeError("could not lay out flat {}".format(flat_id))


def _build(flat_id, turns, rects, furniture, spots, door, dock):
    def xy(point):
        x, y = _turn(turns, *point)
        return round(x, 3), round(y, 3)

    zones = []
    for i, name in enumerate(ZONE_NAMES):
        x0, y0, x1, y1 = _turn_rect(turns, rects[name])
        zone_spots = []
        for s in spots[name]:
            (px, py), (vx, vy) = xy(s["person"]), xy(s["view"])
            zone_spots.append({"name": s["name"], "person_x": px, "person_y": py, "view_x": vx, "view_y": vy,
                               "heading_deg": _wrap(s["heading"] + 90.0 * turns)})
        main = zone_spots[0]
        others = sorted(((_shared_edge(rects[name], rects[o]), o) for o in ZONE_NAMES if o != name), reverse=True)
        zones.append({
            "zone_id": "z{}".format(i + 1), "name": name, "label": LABELS[name], "description": DESCRIPTIONS[name],
            "x": round((x0 + x1) / 2, 3), "y": round((y0 + y1) / 2, 3),
            "size_x": round(x1 - x0, 3), "size_y": round(y1 - y0, 3), "area_m2": round((x1 - x0) * (y1 - y0), 2),
            "furniture": list(ZONE_FURNITURE[name]),
            "view_x": main["view_x"], "view_y": main["view_y"], "heading_deg": main["heading_deg"],
            "person_x": main["person_x"], "person_y": main["person_y"], "spots": zone_spots,
            "neighbours": [o for edge, o in others if edge > 0.5],
        })
    pieces = []
    for name in PIECES:
        f = furniture[name]
        zone = next(z for z, names in ZONE_FURNITURE.items() if name in names)
        x, y = xy((f["x"], f["y"]))
        rect = _turn_rect(turns, f["rect"])
        pieces.append({"name": name, "body": PIECES[name]["body"], "zone": zone, "x": x, "y": y,
                       "yaw_deg": round(_wrap(math.degrees(f["yaw"]) + 90.0 * turns), 1),
                       "size_x": round(rect[2] - rect[0], 3), "size_y": round(rect[3] - rect[1], 3)})
    door_x, door_y = xy(door)
    dock_x, dock_y = xy(dock)
    return {"flat_id": flat_id, "width": round(2 * HALF, 3), "depth": round(2 * HALF, 3), "area_m2": AREA,
            "door": {"wall": DOOR_WALLS[turns], "x": door_x, "y": door_y, "width": DOOR_WIDTH},
            "dock": {"x": dock_x, "y": dock_y}, "zones": zones, "furniture": pieces}


# --- helpers used by the generator, the prior, the simulator and the app follow-up ---

def zone_names(flat):
    return [z["name"] for z in flat["zones"]]


def zone(flat, name):
    for z in flat["zones"]:
        if z["name"] == name:
            return z
    raise KeyError(name)


def zone_rect(z):
    return (z["x"] - z["size_x"] / 2, z["y"] - z["size_y"] / 2, z["x"] + z["size_x"] / 2, z["y"] + z["size_y"] / 2)


def furniture_rect(piece):
    return (piece["x"] - piece["size_x"] / 2, piece["y"] - piece["size_y"] / 2,
            piece["x"] + piece["size_x"] / 2, piece["y"] + piece["size_y"] / 2)


def zone_at(flat, x, y):
    """The name of the zone that the point (x, y) of the floor is in (the nearest one for a point on a border)."""
    return min(flat["zones"], key=lambda z: _point_rect_dist(x, y, zone_rect(z)))["name"]


def signature(flat):
    """The flat's zone set as a string, e.g. 'bedroom,living_room,dining_area,kitchen,hallway'."""
    return ",".join(zone_names(flat))


def view_xy(flat, name):
    if name == "dock":
        return flat["dock"]["x"], flat["dock"]["y"]
    z = zone(flat, name)
    return z["view_x"], z["view_y"]


def distance(flat, a, b):
    """Metres in a straight line between two viewing poses (or 'dock'). Furniture may be in the way: the real
    walk is a little longer."""
    (ax, ay), (bx, by) = view_xy(flat, a), view_xy(flat, b)
    return math.hypot(ax - bx, ay - by)


def bearing_from_dock(flat, x, y):
    """Degrees from the dock to (x, y), 0 = +x, counter-clockwise, in [0, 360)."""
    return math.degrees(math.atan2(y - flat["dock"]["y"], x - flat["dock"]["x"])) % 360.0


def to_app_zones(flat):
    """The flat in the shape of app/knock.py's ZONES: name -> (label, (x, y), heading_deg, description)."""
    return OrderedDict((z["name"], (z["label"], (z["view_x"], z["view_y"]), z["heading_deg"], z["description"]))
                       for z in flat["zones"])


def arena_layout(flat, person_zone="bedroom"):
    """The flat as the layout `Arena.apply()` takes (vision/scene.py): the furniture (and clutter) where the flat
    has it, and the red target pillar standing in for her at her spot in `person_zone`. A flat with clutter needs
    the clutter arena (`Arena(clutter=True)`); `parked_layout` gives the bodies a flat does not list."""
    obstacles = [{"name": f["body"], "x": f["x"], "y": f["y"], "yaw": math.radians(f["yaw_deg"]),
                  "radius": shape(f["name"])["radius"], "color": list(shape(f["name"])["color"])}
                 for f in flat["furniture"]]
    obstacles.sort(key=lambda o: o["name"])
    z = zone(flat, person_zone)
    return {"obstacles": obstacles, "target": (z["person_x"], z["person_y"])}


# --- random obstacles: the app's mess levels (Clean, Medium, Messy) ---
#
# The loose pieces (stool, armchair, crate, plant) and the clutter (sim/clutter.xml: twelve small household things)
# are dropped at random spots on the open floor; the fixed pieces (bed, couch, cabinet, table), the zones, the door
# and the dock stay. A flat's "furniture" lists the 8 pieces always (a loose piece taken out of the room is there
# with parked=True, at its PARK spot outside the walls) plus the clutter that is on the floor (clutter=True). Every
# body a flat does not list is parked (app/robot.py ApartmentScene.load, homes/sim_bridge.FlatEnv).

LOOSE = ("stool", "armchair", "crate", "plant")  # pieces light enough to end up anywhere; the rest stay
PASSAGE = 0.50  # metres between a dropped piece and every other piece and wall (vision/scene.py PASSAGE, the training rule)
SPOT_GAP = 0.50  # between a dropped piece and every viewing spot, the dock and the doorway
PERSON_SPOT_GAP = 0.30  # between a dropped piece and the places where she may lie
REACH_CELL = 0.05  # metres per cell of the reachability check
REACH_INFLATE = 0.30  # the robot's centre keeps this far from furniture and walls (CLEARANCE)
ROUTE_MARGIN = 0.42  # app/robot.py ROUTE_MARGIN (the app's A* map): scatter() keeps every spot reachable with it

# The clutter of sim/clutter.xml (keep the two in step). shape: "box" (half: half its length and width) or
# "circle" (half: its radius twice); radius: the circle around the footprint (as vision/scene.py computes it);
# color: the XML's color (household tones, never red).
CLUTTER = OrderedDict((
    ("laundry_basket", dict(body="obs_c0", shape="circle", half=(0.2, 0.2), radius=0.2, color=(0.84, 0.78, 0.62))),
    ("bin", dict(body="obs_c1", shape="circle", half=(0.13, 0.13), radius=0.13, color=(0.46, 0.5, 0.55))),
    ("cardboard_box", dict(body="obs_c2", shape="box", half=(0.2, 0.15), radius=0.25, color=(0.72, 0.57, 0.38))),
    ("books", dict(body="obs_c3", shape="box", half=(0.13, 0.1), radius=0.164, color=(0.3, 0.38, 0.58))),
    ("pouf", dict(body="obs_c4", shape="circle", half=(0.2, 0.2), radius=0.2, color=(0.55, 0.6, 0.45))),
    ("toy_crate", dict(body="obs_c5", shape="box", half=(0.2, 0.15), radius=0.25, color=(0.92, 0.78, 0.3))),
    ("suitcase", dict(body="obs_c6", shape="box", half=(0.22, 0.12), radius=0.2506, color=(0.26, 0.3, 0.36))),
    ("gym_bag", dict(body="obs_c7", shape="box", half=(0.22, 0.12), radius=0.2506, color=(0.3, 0.5, 0.46))),
    ("vacuum", dict(body="obs_c8", shape="circle", half=(0.15, 0.15), radius=0.15, color=(0.6, 0.55, 0.7))),
    ("bucket", dict(body="obs_c9", shape="circle", half=(0.14, 0.14), radius=0.14, color=(0.36, 0.6, 0.75))),
    ("moving_boxes", dict(body="obs_c10", shape="box", half=(0.17, 0.17), radius=0.2404, color=(0.66, 0.53, 0.37))),
    ("toy_chest", dict(body="obs_c11", shape="box", half=(0.2, 0.12), radius=0.2332, color=(0.5, 0.4, 0.3))),
))
# How many loose things each level puts on the floor: the loose pieces (all or none), clutter wanted and the least
# accepted (messy places as many of the 12 as fit, at least 10), and the passage kept by the clutter.
LEVELS = OrderedDict((
    ("clean", dict(pieces=False, clutter=0, at_least=0, passage=PASSAGE)),
    ("medium", dict(pieces=True, clutter=4, at_least=4, passage=0.50)),
    ("messy", dict(pieces=True, clutter=12, at_least=10, passage=0.40)),
))
PARK_Y = 30.0  # parked bodies wait on a row outside the room (sim/clutter.xml: obs_c<i> at x = 20 + 2i)
PARK = dict([(c["body"], (20.0 + 2.0 * i, PARK_Y)) for i, c in enumerate(CLUTTER.values())]
            + [(PIECES[n]["body"], (46.0 + 2.0 * k, PARK_Y)) for k, n in enumerate(LOOSE)])


def shape(name):
    """The table entry of a piece or a clutter item: PIECES[name] (a box) or CLUTTER[name]."""
    if name in PIECES:
        return dict(PIECES[name], shape="box")
    return CLUTTER[name]


def on_floor(flat):
    """The pieces and clutter of a flat that are in the room (not parked)."""
    return [p for p in flat["furniture"] if not p.get("parked")]


def loose_count(flat):
    """How many loose things are on the floor: loose pieces not parked, plus clutter."""
    return sum(1 for p in on_floor(flat) if p["name"] in LOOSE or p.get("clutter"))


def parked_layout(flat, bodies):
    """Arena.apply() entries that park every body of `bodies` (obs_ names) the flat does not list."""
    listed = {p["body"] for p in flat["furniture"]}
    by_body = dict([(c["body"], c) for c in CLUTTER.values()] + [(PIECES[n]["body"], PIECES[n]) for n in PIECES])
    return [{"name": b, "x": PARK[b][0], "y": PARK[b][1], "yaw": 0.0, "radius": by_body[b]["radius"],
             "color": list(by_body[b]["color"])} for b in bodies if b not in listed and b in PARK]


def _corners(x, y, yaw, half):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in ((half[0], half[1]), (-half[0], half[1]),
                                                                 (-half[0], -half[1]), (half[0], -half[1]))]


def _octagon(x, y, r):
    """An octagon around a circle of radius r (its sides touch the circle): a round item for the polygon tests."""
    out = r / math.cos(math.pi / 8)
    return [(x + out * math.cos(math.pi / 8 + k * math.pi / 4), y + out * math.sin(math.pi / 8 + k * math.pi / 4))
            for k in range(8)]


def _outline(name, x, y, yaw):
    s = shape(name)
    return _octagon(x, y, s["half"][0]) if s["shape"] == "circle" else _corners(x, y, yaw, s["half"])


def _poly_overlap(p, q):
    """Separating-axis test for two convex polygons (lists of corners)."""
    for poly in (p, q):
        for i in range(len(poly)):
            (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
            nx, ny = ay - by, bx - ax
            pa = [nx * x + ny * y for x, y in p]
            qa = [nx * x + ny * y for x, y in q]
            if max(pa) < min(qa) or max(qa) < min(pa):
                return False
    return True


def _seg_point(px, py, a, b):
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy or 1.0)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _poly_point_dist(poly, px, py):
    if _poly_overlap(poly, [(px, py)] * 3):
        return 0.0
    return min(_seg_point(px, py, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def _poly_dist(p, q):
    """Distance between two convex polygons; 0 when they overlap."""
    if _poly_overlap(p, q):
        return 0.0
    return min(min(_poly_point_dist(q, x, y) for x, y in p), min(_poly_point_dist(p, x, y) for x, y in q))


def piece_corners(piece):
    """The floor outline of a piece or clutter item of a flat (its own size, turned by its heading): four corners,
    or eight for a round item."""
    return _outline(piece["name"], piece["x"], piece["y"], math.radians(piece["yaw_deg"]))


def reachable_spots(flat, inflate=REACH_INFLATE, cell=REACH_CELL, spot_radius=0.25):
    """{spot name: True if the robot's centre can get from the dock to within `spot_radius` of it}, for every
    viewing spot of every zone, on a grid that keeps `inflate` metres from furniture, clutter and walls (as the
    planner of nav/dagger/expert.py does). Parked pieces are left out. Needs only numpy."""
    import numpy as np
    from collections import deque
    n = int(math.ceil(2 * HALF / cell))
    xs = -HALF + cell * (np.arange(n) + 0.5)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    clearance = np.minimum(HALF - np.abs(gx), HALF - np.abs(gy))
    for piece in on_floor(flat):
        s = shape(piece["name"])
        dx, dy = gx - piece["x"], gy - piece["y"]
        if s["shape"] == "circle":
            clearance = np.minimum(clearance, np.maximum(np.hypot(dx, dy) - s["half"][0], 0))
            continue
        c, sn = math.cos(math.radians(piece["yaw_deg"])), math.sin(math.radians(piece["yaw_deg"]))
        hx, hy = s["half"]
        lx, ly = c * dx + sn * dy, -sn * dx + c * dy
        clearance = np.minimum(clearance, np.hypot(np.maximum(np.abs(lx) - hx, 0), np.maximum(np.abs(ly) - hy, 0)))
    free = clearance >= inflate
    targets = {"dock": (flat["dock"]["x"], flat["dock"]["y"])}
    for z in flat["zones"]:
        for s in z["spots"]:
            targets["{}:{}".format(z["name"], s["name"])] = (s["view_x"], s["view_y"])
    for x, y in targets.values():  # a small disc around every spot is open (not inside a real shape)
        free |= (np.hypot(gx - x, gy - y) < spot_radius) & (clearance > 0.02)
    start = (int(np.clip((targets["dock"][0] + HALF) // cell, 0, n - 1)),
             int(np.clip((targets["dock"][1] + HALF) // cell, 0, n - 1)))
    seen = np.zeros_like(free)
    if free[start]:
        seen[start] = True
        queue = deque([start])
        while queue:
            i, j = queue.popleft()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < n and 0 <= b < n and free[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    queue.append((a, b))
    out = {}
    for name, (x, y) in targets.items():
        if name == "dock":
            continue
        near = (np.hypot(gx - x, gy - y) < spot_radius) & seen
        out[name] = bool(near.any())
    return out


def _drop(rng, name, placed, points, people, doorway, passage, tries=300):
    """One random spot and heading for `name` on the open floor that keeps every rule, or None."""
    reach = math.hypot(*shape(name)["half"])
    limit = HALF - passage
    for _ in range(tries):
        x, y = (float(v) for v in rng.uniform(-limit + 0.05, limit - 0.05, size=2))
        yaw = float(rng.uniform(-math.pi, math.pi))
        corners = _outline(name, x, y, yaw)
        if max(max(abs(cx), abs(cy)) for cx, cy in corners) > limit:
            continue
        if any(math.hypot(px - x, py - y) < reach + SPOT_GAP and _poly_point_dist(corners, px, py) < SPOT_GAP
               for px, py in points):
            continue
        if any(_poly_point_dist(corners, px, py) < PERSON_SPOT_GAP for px, py in people):
            continue
        if _poly_dist(corners, doorway) < SPOT_GAP:
            continue
        if any(_poly_dist(corners, other) < passage for other in placed):
            continue
        return x, y, yaw, corners
    return None


def _set(p, x, y, yaw, corners):
    xs_, ys_ = [c[0] for c in corners], [c[1] for c in corners]
    p.update(x=round(x, 3), y=round(y, 3), yaw_deg=round(_wrap(math.degrees(yaw)), 1),
             size_x=round(max(xs_) - min(xs_), 3), size_y=round(max(ys_) - min(ys_), 3), moved=True)


def scatter(flat, rng, level="medium", tries=400):
    """The same flat at a mess level (LEVELS):

        clean    nothing loose on the floor: the loose pieces (stool, armchair, crate, plant) are parked outside,
                 only the fixed furniture (bed, couch, cabinet, table) stays
        medium   the 4 loose pieces dropped at random spots on the open floor, plus 4 clutter items (8 loose things)
        messy    the 4 loose pieces plus as many of the 12 clutter items as fit, at least 10 (up to 16 loose things)

    Zones, the fixed pieces, the door and the dock stay. Each dropped thing keeps a passage to every other piece and
    wall (PASSAGE, 0.5 m, the training rule, for the pieces; LEVELS' passage for the clutter: 0.5 m in medium, 0.4 m
    in messy), SPOT_GAP to every viewing spot, the dock and the doorway, and PERSON_SPOT_GAP to where she may lie;
    every viewing spot stays reachable from the dock on the app's A* map (reachable_spots with ROUTE_MARGIN). A
    zone's spot in front of a loose piece (the living room's armchair) goes with it. Any heading. `rng`: a numpy
    Generator (same state, same result). Raises RuntimeError when no arrangement is found."""
    import copy
    if level not in LEVELS:
        raise ValueError("level must be one of: " + ", ".join(LEVELS))
    rule = LEVELS[level]
    base = copy.deepcopy(flat)
    base["furniture"] = [p for p in base["furniture"] if p["name"] in PIECES]  # an already scattered flat: as built
    fixed = [p for p in base["furniture"] if p["name"] not in LOOSE]
    door = base["door"]
    # The doorway: the door's width along its wall, 0.8 m into the room (kept free, as in _try_layout)
    inward = {"xmin": (1.0, 0.0), "xmax": (-1.0, 0.0), "ymin": (0.0, 1.0), "ymax": (0.0, -1.0)}[door["wall"]]
    along = (abs(inward[1]), abs(inward[0]))
    dw = DOOR_WIDTH / 2
    doorway = [(door["x"] + along[0] * a + inward[0] * b, door["y"] + along[1] * a + inward[1] * b)
               for a, b in ((-dw, 0.0), (dw, 0.0), (dw, 0.8), (-dw, 0.8))]
    keep_spots = [s for z in base["zones"] for s in z["spots"] if s["name"] not in LOOSE]
    points = [(s["view_x"], s["view_y"]) for s in keep_spots] + [(base["dock"]["x"], base["dock"]["y"])]
    people = [(s["person_x"], s["person_y"]) for s in keep_spots]
    best = None
    for attempt in range(tries):
        placed = [piece_corners(p) for p in fixed]
        moved = []
        if rule["pieces"]:
            for name in LOOSE:
                spot = _drop(rng, name, placed, points, people, doorway, PASSAGE)
                if spot is None:
                    break
                placed.append(spot[3])
                moved.append((name,) + spot)
            if len(moved) != len(LOOSE):
                continue
        clutter = []
        for k in rng.permutation(len(CLUTTER)):  # which items, and which first, at random
            name = list(CLUTTER)[int(k)]
            if len(clutter) == rule["clutter"]:
                break
            spot = _drop(rng, name, placed, points, people, doorway, rule["passage"], tries=150)
            if spot is not None:
                placed.append(spot[3])
                clutter.append((name,) + spot)
        if len(clutter) < rule["at_least"]:
            continue
        out = copy.deepcopy(base)
        by_name = {p["name"]: p for p in out["furniture"]}
        for name, x, y, yaw, corners in moved:
            _set(by_name[name], x, y, yaw, corners)
            by_name[name]["zone"] = zone_at(out, by_name[name]["x"], by_name[name]["y"])
        if not rule["pieces"]:
            for name in LOOSE:
                p = by_name[name]
                p.update(x=PARK[p["body"]][0], y=PARK[p["body"]][1], yaw_deg=0.0, moved=True, parked=True, zone=None)
        for name, x, y, yaw, corners in clutter:
            p = {"name": name, "body": CLUTTER[name]["body"], "clutter": True}
            _set(p, x, y, yaw, corners)
            p["zone"] = zone_at(out, p["x"], p["y"])
            out["furniture"].append(p)
        for z in out["zones"]:
            z["spots"] = [s for s in z["spots"] if s["name"] not in LOOSE]
            z["furniture"] = [p["name"] for p in out["furniture"] if p["zone"] == z["name"]]
        if not all(reachable_spots(out, inflate=ROUTE_MARGIN).values()):
            continue
        out["scattered"] = True
        out["level"] = level
        if len(clutter) == rule["clutter"]:
            return out
        if best is None or len(clutter) > len([p for p in best["furniture"] if p.get("clutter")]):
            best = out
        if attempt >= 40:  # messy: as many as fit after a fair number of draws
            break
    if best is not None:
        return best
    raise RuntimeError("could not make flat {} {}".format(flat.get("flat_id"), level))
