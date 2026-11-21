"""Show that the Bayesian search works: numbers against simple strategies, and worked examples.

    uv run --group ml python homes/search_demo.py           # 600 residents, 4 weeks of journal -> data/search_examples/
    uv run --group ml python homes/search_demo.py --tune    # find search.WEIGHTS on the train residents

Makes a small population in memory (generate.build), takes the population shares from the train residents, and
runs every test check-in of the test residents (people and flats never seen): journal + habits + floor plan ->
belief -> search order -> metres walked until she is found. No TabPFN: the belief is counting (search.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np

import search as S  # noqa: E402
import search_figure  # noqa: E402
from flats import CLASSES, LABELS  # noqa: E402
from generate import build  # noqa: E402
from residents import SLOTS, stated_habits, zone_probs  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
NAMES = dict(LABELS, out="Out")
STRATEGIES = (
    ("random", "Random order"),
    ("nearest", "Nearest zone first (floor plan only)"),
    ("last", "Last heard first, then nearest"),
    ("population", "Most likely first: population only"),
    ("habits", "Most likely first: + stated habits"),
    ("journal", "Most likely first: + journal"),
    ("full", "Most likely first: + last heard  (the search)"),
    ("oracle", "Most likely first: her true routine (best possible)"),
    ("shortest", "Comparison: full belief, fewest expected metres"),
)
DAYS_DEFAULT = 30  # 4 weeks of journal (days 0-27), then 2 test days


def clock(hour):
    return "{:02d}:{:02d}".format(int(hour), int(round((hour % 1) * 60)) % 60)


def same_part_of_today(hour, last):
    """True when her newest record is from this same part of today (so she has most likely not moved)."""
    return last is not None and last[1] < 16 * 60 and S.slot_of(int(hour * 60 - last[1])) == S.slot_of(int(hour * 60))


def check_ins(out, split):
    """(flat, persona, habits, her diary, her journal, check-in row, last heard) for every check-in of a split."""
    diary = dict(tuple(out["diary"].groupby("resident_id")))
    checkins = dict(tuple(out["checkins"][out["checkins"]["kind"] == "checkin"].groupby("resident_id")))
    for flat, p in zip(out["flats"], out["personas"]):
        if p["resident_split"] != split:
            continue
        mine = diary[p["resident_id"]]
        journal = list(zip(mine["weekday"].astype(int), mine["slot"].astype(int), mine["zone"]))
        for r in checkins[p["resident_id"]].itertuples():
            last = (r.last_heard_zone, float(r.min_since_heard)) if isinstance(r.last_heard_zone, str) else None
            yield flat, p, stated_habits(p), mine, journal, r, last


def tune(out, population, residents=150):
    """The weights with the lowest log loss of the final belief on the check-ins of train residents."""
    rows = [c for c in check_ins(out, "train") if c[1]["resident_id"] <= residents * 5 // 4]
    best = None
    for pop in (4.0, 8.0, 16.0, 32.0, 64.0):
        for habit in (0.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            for outing in (8.0, 16.0, 32.0, 64.0, 128.0, 256.0):
                w = {"population": pop, "habit": habit, "outing": outing}
                loss = -np.mean([np.log(max(S.beliefs(flat, r.hour, int(r.weekday), population, habits, journal, last,
                                                      weights=w)["last_heard"][r.zone], 1e-6))
                                 for flat, p, habits, mine, journal, r, last in rows])
                if best is None or loss < best[0]:
                    best = (loss, w)
    return best


def cases(out, population):
    """One dict per test check-in of the test residents, with the belief and every strategy's walk."""
    rng = np.random.default_rng(0)
    rows = []
    for flat, p, habits, mine, journal, r, last in check_ins(out, "test"):
        b = S.beliefs(flat, r.hour, int(r.weekday), population, habits, journal, last)
        b["oracle"] = zone_probs(p, int(r.weekday), int(r.slot))  # the routine she really draws from (not known)
        orders = {"random": [flat["zones"][i]["name"] for i in rng.permutation(len(flat["zones"]))],
                  "nearest": S.nearest_first(flat), "last": S.nearest_first(flat, last[0] if last else None),
                  "population": S.plan(flat, b["population"])[0], "habits": S.plan(flat, b["habits"])[0],
                  "journal": S.plan(flat, b["journal"])[0], "full": S.plan(flat, b["last_heard"])[0],
                  "shortest": S.plan_shortest(flat, b["last_heard"])[0], "oracle": S.plan(flat, b["oracle"])[0]}
        rows.append({"flat": flat, "persona": p, "habits": habits, "diary": mine, "hour": float(r.hour),
                     "weekday": int(r.weekday), "slot": int(r.slot), "day": int(r.day), "last": last,
                     "truth": r.zone, "beliefs": b, "orders": orders, "recent": same_part_of_today(r.hour, last),
                     "here": (float(r.x), float(r.y)),
                     "walks": {k: S.walk(flat, o, r.zone) for k, o in orders.items()}})
    return rows


BELIEF_OF = {"population": "population", "habits": "habits", "journal": "journal", "full": "last_heard",
             "oracle": "oracle"}


def table(rows):
    """Text and numbers: per strategy, the metres walked until she is found (check-ins with her at home), and for
    the belief rows how good the belief itself is (all check-ins: right most-likely class, log loss)."""
    groups = (("all", "All check-ins", rows),
              ("not_heard_yet", "Not heard yet in this part of the day (habits and journal must do the work)",
               [r for r in rows if not r["recent"]]),
              ("heard_recently", "Already heard in this part of the day", [r for r in rows if r["recent"]]))
    lines, result = [], {}
    for key, title, group in groups:
        home = [r for r in group if r["truth"] != "out"]
        lines += ["", "{} — {} check-ins, {} with her at home".format(title, len(group), len(home)),
                  "{:<54}{:>15}{:>9}{:>12}{:>10}".format("Strategy", "zones checked", "metres", "belief top", "log loss")]
        result[key] = {"check_ins": len(group), "at_home": len(home)}
        for name, label in STRATEGIES:
            metres = float(np.mean([r["walks"][name][0] for r in home]))
            checked = float(np.mean([r["walks"][name][1] for r in home]))
            entry = {"metres": round(metres, 2), "zones_checked": round(checked, 2)}
            tail = "{:>12}{:>10}".format("", "")
            if name in BELIEF_OF:
                stage = BELIEF_OF[name]
                hit = float(np.mean([max(r["beliefs"][stage], key=r["beliefs"][stage].get) == r["truth"] for r in group]))
                loss = float(-np.mean([np.log(max(r["beliefs"][stage][r["truth"]], 1e-6)) for r in group]))
                entry.update(belief_top1=round(hit, 3), log_loss=round(loss, 3))
                tail = "{:>11.0f}%{:>10.2f}".format(100 * hit, loss)
            result[key][name] = entry
            lines.append("{:<54}{:>15.2f}{:>9.1f}{}".format(label, checked, metres, tail))
    said = np.array([r["beliefs"]["last_heard"]["out"] > 0.5 for r in rows])
    is_out = np.array([r["truth"] == "out" for r in rows])
    result["out"] = {"share_out": round(float(is_out.mean()), 3),
                     "caught": round(float(said[is_out].mean()), 3) if is_out.any() else None,
                     "right_when_said": round(float(is_out[said].mean()), 3) if said.any() else None}
    result["residents"] = len({r["persona"]["resident_id"] for r in rows})
    return lines, result


# --- examples ---

def top(probs, home_only=True):
    return max((c for c in CLASSES if not (home_only and c == "out")), key=lambda c: probs[c])


def pick(rows):
    """Six representative check-ins, each from a different resident."""
    def clear(r, least=0.25):  # the first stop is a real guess, not a 2% look on the way
        return r["beliefs"]["last_heard"][r["orders"]["full"][0]] >= least and r["beliefs"]["last_heard"]["out"] < 0.2

    def far(r):  # the search does not simply start at the nearest zone
        return r["orders"]["full"][0] != r["orders"]["nearest"][0]

    def final(r):
        return r["beliefs"]["last_heard"]

    wanted = (
        ("The stated habits point the way",
         lambda r: not r["recent"] and clear(r) and far(r) and r["truth"] == r["orders"]["full"][0]
         and top(r["beliefs"]["population"]) != r["truth"] and top(r["beliefs"]["habits"]) == r["truth"]),
        ("The journal overrules the stated habit",
         lambda r: not r["recent"] and clear(r) and r["truth"] == r["orders"]["full"][0]
         and top(r["beliefs"]["habits"]) != r["truth"] and top(r["beliefs"]["journal"]) == r["truth"]),
        ("Heard a moment ago",
         lambda r: r["recent"] and clear(r, 0.6) and r["last"][0] == r["truth"] == r["orders"]["full"][0]
         and top(r["beliefs"]["journal"]) != r["truth"]),
        ("Last heard is old news",
         lambda r: not r["recent"] and r["last"] is not None and r["last"][0] not in ("out", r["truth"])
         and clear(r, 0.35) and r["truth"] == r["orders"]["full"][0] and r["last"][1] > 120),
        ("First guess wrong, then the belief is updated",
         lambda r: not r["recent"] and clear(r, 0.35) and r["orders"]["full"][0] == top(final(r))
         and r["truth"] == r["orders"]["full"][1] and final(r)[r["truth"]] > 0.15),
        ("Probably not home",
         lambda r: r["truth"] == "out" and final(r)["out"] > 0.6 and not r["recent"]),
    )
    chosen, used = [], set()
    for title, test in wanted:
        for r in rows:
            if r["persona"]["name"] not in used and test(r):  # a different name in every example
                chosen.append((title, r))
                used.add(r["persona"]["name"])
                break
    return chosen


def story(title, r):
    """One sentence that says what the example shows, in numbers."""
    name, b, flat = r["persona"]["name"], r["beliefs"], r["flat"]
    wk = "1" if r["weekday"] >= 5 else "0"
    part = "{} {}s".format("weekend" if wk == "1" else "weekday", SLOTS[r["slot"]])
    habit = r["habits"]["usual"][wk][r["slot"]]
    order, truth = r["orders"]["full"], r["truth"]
    pct = lambda stage, z: "{:.0f}%".format(100 * b[stage][z])  # noqa: E731
    low = lambda z: NAMES[z].lower()  # noqa: E731
    if title.startswith("The stated habits"):
        return ("The form says {} on {}, where most people are in the {}. The search trusts the form, "
                "goes to the {} first and finds {}.".format(low(habit), part, low(top(b["population"])), low(order[0]),
                                                            name))
    if title.startswith("The journal"):
        d = r["diary"]
        same = d[(d["slot"] == r["slot"]) & ((d["weekday"] >= 5) == (wk == "1"))]
        n = int((same["zone"] == habit).sum())
        return ("The form says {}, but {} was there on only {} of {} {}. The journal lowers the {} from {} to {}; "
                "{} is in the {}.".format(low(habit), name, n, len(same), part, low(habit), pct("habits", habit),
                                          pct("journal", habit), name, low(truth)))
    if title.startswith("Heard"):
        return ("{} was heard in the {} {} ago, in this same part of the day, so the {} jumps from {} to {}."
                .format(name, low(r["last"][0]), search_figure.ago(r["last"][1]), low(r["last"][0]),
                        pct("journal", r["last"][0]), pct("last_heard", r["last"][0])))
    if title.startswith("Last heard"):
        return ("{} was last heard in the {}, but {} ago, in an earlier part of the day. Bayes' rule gives that little "
                "weight; the search goes to the most likely zone, the {} ({}), and finds {} there.".format(
                    name, low(r["last"][0]), search_figure.ago(r["last"][1]), low(order[0]), pct("last_heard", order[0]),
                    name))
    if title.startswith("First guess"):
        after = S.after_miss(b["last_heard"], order[0])
        return ("The {} is the best bet ({}), but {} is not there. Its share moves to the rest: the {} rises "
                "from {} to {:.0f}%, and {} is there.".format(low(order[0]), pct("last_heard", order[0]), name,
                                                              low(order[1]), pct("last_heard", order[1]),
                                                              100 * after[order[1]], name))
    return ("{}'s form lists a regular outing on {} {}s, so the belief says {} not home. "
            "Knock could phone first instead of walking the whole flat.".format(
                name, DAYS[r["weekday"]], SLOTS[r["slot"]], pct("last_heard", "out")))


def describe(title, r):
    """The example as text: the three inputs, the belief step by step, the search and what happened."""
    p, flat, b = r["persona"], r["flat"], r["beliefs"]
    wk = "1" if r["weekday"] >= 5 else "0"
    lines = [title.upper(), "",
             "{} · {} {} ({})".format(p["name"], DAYS[r["weekday"]], clock(r["hour"]), SLOTS[r["slot"]]), "",
             "HABITS (stated on a form)"]
    for key, label in (("0", "weekdays"), ("1", "weekends")):
        lines.append("  {:<9}".format(label) + " · ".join(
            "{} {}".format(SLOTS[s], NAMES[z].lower()) for s, z in enumerate(r["habits"]["usual"][key])))
    outings = ", ".join("{} {}".format(DAYS[wd], SLOTS[s]) for wd, s in r["habits"]["outings"])
    lines.append("  out      " + (outings or "no regular outings"))
    d = r["diary"]
    n_days = int(d["day"].nunique())
    lines += ["", "JOURNAL: {} days of records (where the robot heard her; 'out' = nobody home)".format(n_days),
              "  last week {:<4}{:<14}{:<14}{:<14}".format("", *SLOTS)]
    for day, g in d[d["day"] >= d["day"].max() - 6].groupby("day"):
        cells = {int(s): NAMES[z].lower() for s, z in zip(g["slot"], g["zone"])}
        lines.append("  {:<13}{:<14}{:<14}{:<14}".format(DAYS[int(g["weekday"].iloc[0])], *[cells.get(s, "") for s in range(3)]))
    same = d[(d["slot"] == r["slot"]) & ((d["weekday"] >= 5) == (r["weekday"] >= 5))]
    counts = same["zone"].value_counts()
    lines.append("  all {} {}s: ".format("weekend" if r["weekday"] >= 5 else "weekday", SLOTS[r["slot"]])
                 + ", ".join("{} {}".format(NAMES[z].lower(), n) for z, n in counts.items()))
    if r["last"]:
        h, m = divmod(int(round(r["last"][1])), 60)
        lines.append("  last heard: {}, {} ago{}".format(
            NAMES[r["last"][0]].lower(), "{} h {:02d} min".format(h, m) if h else "{} min".format(m),
            " (this {})".format(SLOTS[r["slot"]]) if r["recent"] else ""))
    lines += ["", "BELIEF, step by step",
              "  {:<13}{:>11}{:>9}{:>10}{:>12}".format("", "population", "+habits", "+journal", "+last heard")]
    for c in CLASSES:
        lines.append("  {:<13}{:>10.0f}%{:>8.0f}%{:>9.0f}%{:>11.0f}%".format(
            NAMES[c], *[100 * b[s][c] for s in S.STAGES]))
    order = r["orders"]["full"]
    lines += ["", "SEARCH ORDER (most likely zone first)"]
    probs, metres, at = dict(b["last_heard"]), 0.0, "dock"
    names, dist = S.distances(flat)
    for n, name in enumerate(order, start=1):
        metres += float(dist[names.index(at), names.index(name)])
        found = name == r["truth"]
        lines.append("  {}. {:<12}{:>4.0f}%  after {:>4.1f} m{}".format(
            n, NAMES[name], 100 * probs[name], metres, "   <- found her" if found else "   not there"))
        if found:
            break
        probs, at = S.after_miss(probs, name), name
    if r["truth"] == "out":
        lines.append("  Nobody home. The belief said out {:.0f}% before the robot started.".format(
            100 * b["last_heard"]["out"]))
    lines += ["", "Zones checked until found: this search {} · nearest first {} · last heard first {}".format(
        r["walks"]["full"][1], r["walks"]["nearest"][1], r["walks"]["last"][1])]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--residents", type=int, default=600)
    parser.add_argument("--days", type=int, default=DAYS_DEFAULT, help="days simulated; the last 2 are test days")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--tune", action="store_true", help="find the weights for search.WEIGHTS (train residents) and stop")
    args = parser.parse_args()
    out = build(residents=args.residents, days=args.days, seed=args.seed)
    population = S.population_shares(out["diary"])
    if args.tune:
        loss, weights = tune(out, population)
        print("lowest log loss on train check-ins {:.3f} with {}".format(loss, weights))
        return
    rows = cases(out, population)
    lines, result = table(rows)
    print("{} test residents (people and flats never seen), {} days of journal each. Metres: walked from the dock "
          "until she is found.".format(result["residents"], args.days - 2))
    print("\n".join(lines))
    print("\nShe was out in {:.0f}% of check-ins. The belief said 'out' (over 50%) for {:.0f}% of those, "
          "and was right {:.0f}% of the times it said so.".format(
              100 * result["out"]["share_out"], 100 * result["out"]["caught"], 100 * result["out"]["right_when_said"]))
    data = Path(args.out)
    examples = data / "search_examples"
    examples.mkdir(parents=True, exist_ok=True)
    for old in examples.glob("example_*.png"):
        old.unlink()
    for i, (title, r) in enumerate(pick(rows), start=1):
        search_figure.figure(title, story(title, r), r, examples / "example_{}.png".format(i))
        print("\n" + "=" * 100 + "\n" + story(title, r) + "\n\n" + "\n".join(describe(title, r)))
    (data / "search_results.json").write_text(json.dumps(
        dict(result, residents_generated=args.residents, days=args.days, seed=args.seed, weights=S.WEIGHTS), indent=1))
    print("\nwrote", examples, "and", data / "search_results.json")


if __name__ == "__main__":
    main()
