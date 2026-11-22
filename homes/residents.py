"""Random residents as a small weekly routine table, no timeline.

A resident has 3 slots a day (morning 07:00-12:00, afternoon 12:00-18:00, evening 18:00-22:30) and, for weekdays
and for weekends, a probability over the zones of her flat for each slot, plus a probability of being out. The table
is the routine and the deviation at once: if she is in the living room in 80% of her evenings, the table says 0.8.
Each day she draws one zone per slot. Night (before 07:00 and after 22:30) she is in the bedroom.
Every flat has the same five zones (flats.py), so a table always covers all of them.

Eight flavours (`archetype`) give the base tables; every resident gets her own jittered copy. Needs only numpy.
"""

import numpy as np

from flats import CLASSES, ZONE_NAMES

CODE = {name: i for i, name in enumerate(CLASSES)}
OUT = CODE["out"]
SLOTS = ("morning", "afternoon", "evening")
SLOT_WINDOWS = ((420, 720), (720, 1080), (1080, 1350))  # minutes of the day
MIX_ZONES = ZONE_NAMES

FIRST_NAMES = (
    "Margarete", "Helga", "Ingrid", "Gisela", "Erika", "Ursula", "Renate", "Brigitte", "Waltraud", "Hildegard",
    "Elfriede", "Irmgard", "Christa", "Monika", "Gertrud", "Anneliese", "Charlotte", "Lieselotte", "Edith", "Rosemarie",
    "Werner", "Horst", "Klaus", "Gerhard", "Heinz", "Dieter", "Manfred", "Rolf", "Walter", "Karl",
    "Hans", "Jürgen", "Wolfgang", "Günter", "Peter", "Fritz", "Otto", "Siegfried", "Ernst", "Bernd",
    "Lotte", "Hannelore", "Marianne", "Annemarie", "Ilse", "Luise", "Emma", "Frieda", "Gerda", "Else",
)

_STANDARD = (dict(living_room=0.30, dining_area=0.25, kitchen=0.25, bedroom=0.15, hallway=0.05),
             dict(living_room=0.60, bedroom=0.20, dining_area=0.10, kitchen=0.10),
             dict(living_room=0.65, dining_area=0.15, kitchen=0.10, bedroom=0.10))
# weight: how often drawn. mix: home zone mix for (morning, afternoon, evening). conc: how tightly a resident sticks
# to it (low = erratic). shop: (min, max) weekdays with an afternoon outing. daycare: probability of day-care days.
ARCHETYPES = {
    "standard": dict(weight=0.22, mix=_STANDARD, conc=(12, 30), shop=(2, 4), daycare=0.15),
    "early_riser": dict(weight=0.12, mix=(dict(living_room=0.5, dining_area=0.3, kitchen=0.2),
                                          dict(living_room=0.6, bedroom=0.3, dining_area=0.1),
                                          dict(bedroom=0.4, living_room=0.4, dining_area=0.2)),
                        conc=(12, 30), shop=(3, 4), daycare=0.1),
    "night_owl": dict(weight=0.12, mix=(dict(bedroom=0.5, kitchen=0.2, living_room=0.2, dining_area=0.1),
                                        dict(living_room=0.6, kitchen=0.2, dining_area=0.2),
                                        dict(living_room=0.8, dining_area=0.1, kitchen=0.1)),
                      conc=(12, 30), shop=(1, 2), daycare=0.1),
    "kitchen_person": dict(weight=0.13, mix=(dict(kitchen=0.45, dining_area=0.25, living_room=0.3),
                                             dict(kitchen=0.4, dining_area=0.3, living_room=0.3),
                                             dict(living_room=0.55, kitchen=0.2, dining_area=0.15, bedroom=0.1)),
                           conc=(12, 30), shop=(1, 3), daycare=0.1),
    "day_care": dict(weight=0.12, mix=_STANDARD, conc=(12, 30), shop=(1, 2), daycare=1.0),
    "restless": dict(weight=0.10,
                     mix=(dict(living_room=0.3, kitchen=0.2, dining_area=0.15, bedroom=0.2, hallway=0.15),
                          dict(living_room=0.35, kitchen=0.2, dining_area=0.15, bedroom=0.15, hallway=0.15),
                          dict(living_room=0.45, kitchen=0.15, dining_area=0.1, bedroom=0.15, hallway=0.15)),
                     conc=(4, 9), shop=(1, 2), daycare=0.1),
    "frail": dict(weight=0.09, mix=(dict(bedroom=0.5, living_room=0.4, dining_area=0.1),
                                    dict(bedroom=0.55, living_room=0.45),
                                    dict(bedroom=0.5, living_room=0.5)), conc=(12, 30), shop=(0, 1), daycare=0.3),
    "living_room_reader": dict(weight=0.10, mix=(dict(living_room=0.7, dining_area=0.15, kitchen=0.15),
                                                 dict(living_room=0.85, bedroom=0.15),
                                                 dict(living_room=0.85, bedroom=0.1, dining_area=0.05)),
                               conc=(12, 30), shop=(1, 2), daycare=0.1),
}
WEEKEND_MORNING = dict(bedroom=0.5, dining_area=0.5)  # a slow start: part of the weekend morning mix


def _table(rng, base, conc):
    base = dict(base)
    base["hallway"] = base.get("hallway", 0.0) + 0.06  # everyone goes to the door now and then
    alpha = np.array([base.get(z, 0.0) * conc + 0.15 for z in MIX_ZONES])
    w = rng.dirichlet(alpha)
    return {z: float(w[i]) for i, z in enumerate(MIX_ZONES) if w[i] > 0}


def windows(n=3):
    """The n slot windows (start, end minutes) of the 07:00-22:30 day. n = 3 is morning, afternoon, evening."""
    if n == 3:
        return SLOT_WINDOWS
    edges = np.linspace(SLOT_WINDOWS[0][0], SLOT_WINDOWS[-1][1], n + 1)
    return tuple((float(edges[i]), float(edges[i + 1])) for i in range(n))


def sample_persona(rng, idx, flat, n_slots=3):
    """One random resident living in `flat`. `idx` is 0-based; she is called r<idx+1>.

    With n_slots > 3 the day is cut into finer slots; each gets its own draw around the mix of the part of the day it
    falls in (morning, afternoon, evening), so she moves around inside those parts."""
    kinds = list(ARCHETYPES)
    weights = np.array([ARCHETYPES[k]["weight"] for k in kinds])
    arch = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
    A = ARCHETYPES[arch]
    conc = float(rng.uniform(*A["conc"]))
    tables = {"0": [], "1": []}  # "0" weekday, "1" weekend: one zone distribution per slot
    for fine, (a, b) in enumerate(windows(n_slots)):
        slot = slot_of(int((a + b) / 2))  # the part of the day: 0 morning, 1 afternoon, 2 evening
        tables["0"].append(_table(rng, A["mix"][slot], conc))
        base = dict(A["mix"][slot])
        if slot == 0:
            base = {z: 0.7 * base.get(z, 0.0) + 0.3 * WEEKEND_MORNING.get(z, 0.0) for z in MIX_ZONES}
        tables["1"].append(_table(rng, base, conc))

    # Out of the flat: small everyday chance, plus shop afternoons and day-care mornings on chosen weekdays.
    out_p = [[0.01, 0.03, 0.0] for _ in range(7)]
    lo, hi = A["shop"]
    for wd in rng.choice(5, size=int(rng.integers(lo, hi + 1)), replace=False):
        out_p[int(wd)][1] = float(rng.uniform(0.55, 0.9))
    if rng.random() < 0.5:  # a weekend outing
        out_p[int(rng.choice([5, 6]))][0 if rng.random() < 0.6 else 1] = float(rng.uniform(0.4, 0.8))
    if rng.random() < A["daycare"]:
        days = range(5) if arch == "day_care" else rng.choice(5, size=int(rng.integers(1, 3)), replace=False)
        for wd in days:
            out_p[int(wd)][0] = float(rng.uniform(0.85, 0.95))
    part = [slot_of(int((a + b) / 2)) for a, b in windows(n_slots)]
    out_p = [[row[c] for c in part] for row in out_p]
    return {
        "resident_id": idx + 1, "resident": "r{}".format(idx + 1), "name": FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))],
        "archetype": arch, "flat_id": flat["flat_id"], "resident_split": "test" if idx % 5 == 4 else "train",
        "start_weekday": int(rng.integers(7)), "conc": conc, "n_slots": n_slots, "tables": tables, "out_p": out_p,
    }


def slot_of(minute, n=3):
    """Slot index of a minute of the day (0 .. n-1); None at night (before 07:00 or from 22:30)."""
    for s, (a, b) in enumerate(windows(n)):
        if a <= minute < b:
            return s
    return None


def draw_day(p, day, rng):
    """Her zone code in each of the 3 slots on this day (an index into flats.CLASSES; the last one is out)."""
    weekday = (p["start_weekday"] + day) % 7
    table = p["tables"]["1" if weekday >= 5 else "0"]
    codes = []
    for slot in range(len(table)):
        if rng.random() < p["out_p"][weekday][slot]:
            codes.append(OUT)
        else:
            names = list(table[slot])
            probs = np.array([table[slot][n] for n in names])
            codes.append(CODE[names[int(rng.choice(len(names), p=probs / probs.sum()))]])
    return codes


def zone_probs(p, weekday, slot):
    """The routine itself: P(zone) for a weekday and slot, including 'out'. Returns {class name: p}."""
    table = p["tables"]["1" if weekday >= 5 else "0"][slot]
    q = p["out_p"][weekday][slot]
    out = {n: (1 - q) * v for n, v in table.items()}
    out["out"] = q
    return out


STATED_WRONG = 0.3  # share of parts of the day where the stated usual zone is not her real most likely zone


def stated_habits(p, wrong=STATED_WRONG):
    """Her habits as she or her carer would state them on a form: the usual zone for each part of the day (weekdays
    and weekends), and the regular outings as (weekday, slot) pairs. Coarser than her routine table, and imperfect:
    in `wrong` of the parts of the day the form names her second most likely zone (people misjudge their habits).
    The same resident always gets the same form."""
    rng = np.random.default_rng([p["resident_id"], 7])
    usual = {}
    for wk in ("0", "1"):
        usual[wk] = []
        for table in p["tables"][wk]:
            ranked = sorted(table, key=table.get, reverse=True)
            usual[wk].append(ranked[1] if len(ranked) > 1 and rng.random() < wrong else ranked[0])
    outings = [(wd, slot) for wd in range(7) for slot, q in enumerate(p["out_p"][wd]) if q >= 0.5]
    return {"usual": usual, "outings": outings}
