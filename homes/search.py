"""Bayesian search: journal + habits + floor plan -> where to look for her, in which order.

    beliefs(flat, hour, weekday, population, habits, journal, last)   -> P(zone) and P(out), step by step
    plan(flat, probs)                                                  -> the order to check the zones in:
                                                                          most likely first
    after_miss(probs, zone)                                            -> P(zone) once a zone was checked, nobody there

The belief is counting with Bayes' rule, no model to train:

1. Population: how often people are in each zone in this part of the day (weekday or weekend), as a few
   pseudo-records. This is all there is for a new resident.
2. Habits: what she or her carer stated (residents.stated_habits): her usual zone in this part of the day gets
   extra pseudo-records, and so does `out` on the weekday of a regular outing.
3. Journal: every record of her in this part of the day (same weekday type) counts as one record.
4. Last heard: if the newest record is from this same part of today, she has most likely not moved. Bayes' rule
   with the chance that the record was logged in the wrong zone (10%, to a neighbouring zone).

The search goes where she most probably is: `plan` checks the zones from the most to the least likely. Distance does
not change the order (only a tie, within 1 point, goes to the nearer zone). After a miss, `after_miss` gives the
checked zone's probability to the others, so the next zone is again the most likely one. The floor plan defines the
zones, the viewing poses and the path; `walk` measures the metres. `plan_shortest` (fewest expected metres, which can
put a near, less likely zone first) is kept only as a comparison. Needs only numpy.
"""

import itertools

import numpy as np

from flats import CLASSES, ZONE_NAMES, view_xy
from residents import slot_of

# How many journal records each source is worth (pseudo-records). Chosen by `search_demo.py --tune`: the values
# with the lowest log loss on the check-ins of TRAIN residents (counting, not model training).
WEIGHTS = {"population": 8.0, "habit": 2.0, "outing": 32.0}
MISPLACED = 0.1  # share of records logged in a neighbouring zone (HEARD_MISPLACED in generate.py)
MISS_RATE = 0.2  # the mock observation: chance that the robot looks at the zone she is in and does not see her
STOP_OUT = 0.95  # the loop ends with "nobody home" once the belief in `out` reaches this
MAX_LOOKS = 12  # and gives up after this many looks
NIGHT = {"bedroom": 0.85, "kitchen": 0.15}  # no records at night (22:30-07:00): a stated default
STAGES = ("population", "habits", "journal", "last_heard")


def population_shares(diary):
    """P(class | weekday type, slot) from the train residents' diary: array [2, slots, classes]."""
    train = diary[diary["resident_split"] == "train"]
    counts = np.full((2, int(train["slot"].max()) + 1, len(CLASSES)), 0.5)
    code = {c: i for i, c in enumerate(CLASSES)}
    np.add.at(counts, (train["is_weekend"].to_numpy().astype(int), train["slot"].to_numpy().astype(int),
                       train["zone"].map(code).to_numpy()), 1.0)
    return counts / counts.sum(axis=2, keepdims=True)


def _named(p):
    return {c: float(p[i]) for i, c in enumerate(CLASSES)}


def beliefs(flat, hour, weekday, population=None, habits=None, journal=(), last=None, n_slots=3, weights=None):
    """P over the five zones and `out` at this hour (0-24) and weekday (0 = Monday), after each step.

    population: array from population_shares(), or None (every class equally likely).
    habits:     residents.stated_habits(), or None.
    journal:    her records so far, each (weekday, slot, zone).
    last:       (zone, minutes ago) of her newest record, or None.
    Returns {"population": {...}, "habits": {...}, "journal": {...}, "last_heard": {...}}; each is the belief with
    everything up to that step, and "last_heard" is the final one."""
    w = weights or WEIGHTS
    slot = slot_of(int(hour * 60), n_slots)
    code = {c: i for i, c in enumerate(CLASSES)}
    if slot is None:
        night = np.array([NIGHT.get(c, 0.0) for c in CLASSES])
        return {stage: _named(night) for stage in STAGES}
    wk = 1 if weekday >= 5 else 0
    share = population[wk, slot] if population is not None else np.full(len(CLASSES), 1.0 / len(CLASSES))
    counts = w["population"] * share
    out = {"population": _named(counts / counts.sum())}

    outings = set(map(tuple, habits["outings"])) if habits else set()
    if habits:
        counts = counts.copy()
        counts[code[habits["usual"][str(wk)][slot]]] += w["habit"]
        if (weekday, slot) in outings:
            counts[code["out"]] += w["outing"]
    out["habits"] = _named(counts / counts.sum())

    counts = counts.copy()
    for r_weekday, r_slot, r_zone in journal:
        if r_slot != slot or (1 if r_weekday >= 5 else 0) != wk:
            continue
        # A record made during a stated outing says nothing about an ordinary day.
        if r_zone == "out" and (r_weekday, r_slot) in outings and (weekday, slot) not in outings:
            continue
        counts[code[r_zone]] += 1.0
    p = counts / counts.sum()
    out["journal"] = _named(p)

    if last is not None and last[0] is not None and slot_of(int(hour * 60 - last[1]), n_slots) == slot \
            and last[1] < 16 * 60:
        like = np.zeros(len(CLASSES))  # P(the record says last[0] | she is in zone z)
        if last[0] == "out":
            like[code["out"]] = 1.0
        else:
            for z in flat["zones"]:
                if z["name"] == last[0]:
                    like[code[z["name"]]] = 1.0 - MISPLACED
                elif last[0] in z["neighbours"]:
                    like[code[z["name"]]] = MISPLACED / len(z["neighbours"])
        p = p * like
        p = p / p.sum()
    out["last_heard"] = _named(p)
    return out


# --- the floor plan: from belief to search ---

def distances(flat):
    """Straight-line metres between the dock and the zones' viewing poses: (names, matrix); names[0] is 'dock'."""
    names = ["dock"] + [z["name"] for z in flat["zones"]]
    xy = np.array([view_xy(flat, n) for n in names])
    return names, np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])


_ORDERS = np.array(list(itertools.permutations(range(1, len(ZONE_NAMES) + 1))))


def plan(flat, probs):
    """The search order: the most likely zone first, then the next most likely, and so on. A tie (within 1 point)
    goes to the zone nearer to where the robot is. Returns (order, expected metres until she is found)."""
    order, at = [], "dock"
    left = [z["name"] for z in flat["zones"]]
    while left:
        at = next_zone(flat, probs, at, among=left)
        order.append(at)
        left.remove(at)
    return order, expected_metres(flat, order, probs)


def next_zone(flat, probs, at="dock", among=None):
    """Where to look next: the most likely zone under the current belief. A tie (within 1 point) goes to the zone
    nearer to `at`, where the robot is. `among` limits the choice (the zones not checked yet)."""
    names, dist = distances(flat)
    among = list(among) if among is not None else names[1:]
    top = max(probs.get(n, 0.0) for n in among)
    tied = [n for n in among if probs.get(n, 0.0) >= top - 0.01]
    return min(tied, key=lambda n: dist[names.index(at), names.index(n)])


def expected_metres(flat, order, probs):
    """Expected metres walked from the dock until she is found (the whole round if she is out)."""
    total = sum(probs.get(z, 0.0) * walk(flat, order, z)[0] for z in order)
    return total + probs.get("out", 0.0) * walk(flat, order, "out")[0]


def plan_shortest(flat, probs):
    """For comparison only: the order with the fewest expected metres, which may check a near, less likely zone
    first. Returns (order, expected metres)."""
    names, dist = distances(flat)
    legs = dist[np.c_[np.zeros(len(_ORDERS), int), _ORDERS[:, :-1]], _ORDERS]
    walked = np.cumsum(legs, axis=1)  # metres walked when arriving at the i-th zone of each order
    p = np.array([probs.get(n, 0.0) for n in names])
    expected = (walked * p[_ORDERS]).sum(axis=1) + probs.get("out", 0.0) * walked[:, -1]
    best = int(np.argmin(expected))
    return [names[i] for i in _ORDERS[best]], float(expected[best])


def nearest_first(flat, first=None):
    """Always walk to the nearest unchecked zone (after `first`, if given): uses the floor plan but no belief."""
    names, dist = distances(flat)
    order, at = [], 0
    if first in names[1:]:
        order.append(first)
        at = names.index(first)
    while len(order) < len(names) - 1:
        at = min((i for i in range(1, len(names)) if names[i] not in order), key=lambda i: dist[at, i])
        order.append(names[at])
    return order


def walk(flat, order, truth):
    """(metres walked, zones checked) until she is found in `truth`; if she is out, the whole round."""
    names, dist = distances(flat)
    metres, at = 0.0, 0
    for n, name in enumerate(order, start=1):
        i = names.index(name)
        metres, at = metres + float(dist[at, i]), i
        if name == truth:
            return metres, n
    return metres, len(order)


def after_miss(probs, checked, miss_rate=0.0):
    """The belief once `checked` was looked at and she was not seen there (Bayes' rule).

    `miss_rate` is the chance that the robot overlooks her when she is in the zone it looks at. With 0 the zone is
    ruled out and the rest share its part. Above 0 the zone only becomes less likely:
    P(zone | not seen) = P(zone) * miss_rate / (1 - P(zone) * (1 - miss_rate)), and the others grow by the same
    factor 1 / (1 - P(zone) * (1 - miss_rate)). So a zone can become the most likely one again later."""
    seen_chance = probs.get(checked, 0.0) * (1.0 - miss_rate)  # P(she is there and the robot sees her)
    rest = 1.0 - seen_chance
    if rest < 1e-12:  # the belief was sure she is there and the look cannot miss: nothing is left
        return {k: 0.0 for k in probs}
    out = {k: v / rest for k, v in probs.items()}
    out[checked] = probs.get(checked, 0.0) * miss_rate / rest
    return out


# --- the loop: proposal -> observation -> update -> next zone, until found ---

def observe(zone, truth, rng, miss_rate=MISS_RATE):
    """The mock observation: the robot looks at `zone`. True when it sees her. It never sees her where she is not
    (no false alarms); when she is there it overlooks her with probability `miss_rate` (a false negative)."""
    return zone == truth and rng.random() >= miss_rate


class Search:
    """The search loop as an object, so that anything can drive it: the mock in run(), or later the simulator
    (walk the robot to the goal with the navigation controller, run the person detector, report the result).

        search = Search(flat, beliefs(...)["last_heard"])
        while (goal := search.goal()) is not None:      # {"zone", "x", "y", "heading_deg", "p"}: where to go and look
            seen = ...walk to (goal x, y), face heading_deg, look...
            search.report(seen)
        search.status                                   # "found", "nobody_home" or "gave_up"

    `miss_rate` is what the update assumes about the look (how often it overlooks her). `steps` keeps one dict per
    look: zone, p_before, belief_before, seen, belief_after, metres (straight-line metres walked so far)."""

    def __init__(self, flat, probs, miss_rate=MISS_RATE, max_looks=MAX_LOOKS, stop_out=STOP_OUT):
        self.flat, self.belief, self.miss_rate = flat, dict(probs), miss_rate
        self.max_looks, self.stop_out = max_looks, stop_out
        self.names, self.dist = distances(flat)
        self.at, self.metres, self.steps, self.status = "dock", 0.0, [], "searching"
        self._goal = None

    def goal(self):
        """The next place to look: the most likely zone's viewing pose. None when the search is over."""
        if self.status != "searching":
            return None
        open_zones = [n for n in self.names[1:] if self.belief.get(n, 0.0) > 1e-9]
        if self.belief.get("out", 0.0) >= self.stop_out or not open_zones:
            self.status = "nobody_home"
            return None
        if len(self.steps) >= self.max_looks:
            self.status = "gave_up"
            return None
        name = next_zone(self.flat, self.belief, self.at, among=open_zones)
        z = next(z for z in self.flat["zones"] if z["name"] == name)
        self._goal = {"zone": name, "x": z["view_x"], "y": z["view_y"], "heading_deg": z["heading_deg"],
                      "p": self.belief[name]}
        return self._goal

    def report(self, seen):
        """What the robot saw at the goal: True (she is there) or False (nobody seen). Updates the belief."""
        zone = self._goal["zone"]
        self.metres += float(self.dist[self.names.index(self.at), self.names.index(zone)])
        if seen:
            after = {c: (1.0 if c == zone else 0.0) for c in self.belief}
            self.status = "found"
        else:
            after = after_miss(self.belief, zone, self.miss_rate)
        self.steps.append({"zone": zone, "p_before": self.belief[zone], "belief_before": self.belief,
                           "seen": bool(seen), "belief_after": after, "metres": self.metres})
        self.belief, self.at = after, zone

    def report_found(self, zone):
        """She was seen in `zone`, which need not be the goal's (the robot can spot her on the way, or from a
        viewing pose that sees into the next zone). Ends the search as "found" with that zone."""
        if zone not in self.belief:
            raise ValueError("unknown zone {!r}".format(zone))
        self.metres += float(self.dist[self.names.index(self.at), self.names.index(zone)])
        after = {c: (1.0 if c == zone else 0.0) for c in self.belief}
        self.steps.append({"zone": zone, "p_before": self.belief[zone], "belief_before": self.belief,
                           "seen": True, "belief_after": after, "metres": self.metres})
        self.belief, self.at, self.status = after, zone, "found"


def run(flat, probs, truth, rng, miss_rate=MISS_RATE, max_looks=MAX_LOOKS, stop_out=STOP_OUT, update_rate=None):
    """The whole search with the mock observation: pick the most likely zone, walk there, look, update, repeat.

    probs: the starting belief (the final step of beliefs()). truth: the zone she is in, or "out".
    Ends when she is seen, when the belief in `out` reaches `stop_out` (nobody home), or after `max_looks` looks.
    `update_rate` is the miss rate the robot assumes in its update (default: the true `miss_rate`); 0 makes it
    treat every miss as certain, which is the mistake the loop is there to avoid.
    Returns Search.steps: one dict per look."""
    search = Search(flat, probs, miss_rate if update_rate is None else update_rate, max_looks, stop_out)
    while (goal := search.goal()) is not None:
        search.report(observe(goal["zone"], truth, rng, miss_rate))
    return search.steps
