"""Build the DAgger v2 report material from the stored results: python -m nav.dagger.report --out docs/dagger

Writes, from data/nav/dagger/v2 (every number is computed here from the experiments' rows and episodes, not typed):
  experiments.md     every variant tried: its one change, inputs, context, fit time, results on the first 100
                     screening layouts and the paired comparison with its reference (sign test, Wilcoxon)
  promotion.md       the promotion and robust-check runs (200 screening, confirmation, two final seeds, the expert)
  dagger_rounds.md   the DAgger data collections (which student drove, beta, outcomes, rows)
  report_data.json   all of the above as data
  results.png        goal rate with 95% intervals for the main steps
  route-<episode>.png  student path (the trained model driving alone) against the expert's path on the same layout
  compute.md         with --jobs: every job's machine, kind, episodes, steps and hours, summed per machine
Only episodes driven entirely by the student are drawn: `student_only` checks every row.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import dataset_dir, experiment_dir, load_config, set_plan, v2_dir
from .evaluate import compare, episode_table, summarize

# Each variant is compared with the variant it changed (on the same screening layouts).
REFERENCE = {
    "v00-noise": "v03-reference", "v01-rule": "v03-reference", "v02-summary": "v03-reference",
    "v06-knn": "v03-reference", "v09-scan": "v03-reference", "v10-band": "v03-reference", "v11-lidar": "v03-reference",
    "v12-lidar-perturb": "v11-lidar", "v13-lidar-gated": "v11-lidar", "v14-lidar-only": "v11-lidar",
    "v15-lidar-allrows": "v11-lidar", "v16-lidar-history": "v11-lidar", "v17-lidar-balanced": "v11-lidar",
    "v18-lidar-dagger1": "v11-lidar", "v19-lidar-dagger1-near": "v18-lidar-dagger1",
    "v20-lidar-allrows-dagger1": "v15-lidar-allrows", "v21-lidar-allrows-dagger12": "v15-lidar-allrows",
    "v22c-collector": "v15-lidar-allrows", "v23-gated-allrows": "v15-lidar-allrows",
    "v24-allrows-est8": "v15-lidar-allrows", "v27-allrows-lidarshield": "v15-lidar-allrows",
    "v28-allrows-shield2": "v15-lidar-allrows", "v08b-ceiling-lookahead": "v15-lidar-allrows",
    "v29-lidarplan": "v15-lidar-allrows", "v31-lidarplan-alldata": "v29-lidarplan",
}
ORDER = ["v03-reference", "v00-noise", "v01-rule", "v06-knn", "v02-summary", "v09-scan", "v10-band", "v11-lidar",
         "v12-lidar-perturb", "v13-lidar-gated", "v14-lidar-only", "v16-lidar-history", "v17-lidar-balanced",
         "v15-lidar-allrows", "v18-lidar-dagger1", "v19-lidar-dagger1-near", "v20-lidar-allrows-dagger1",
         "v21-lidar-allrows-dagger12", "v22c-collector", "v23-gated-allrows", "v24-allrows-est8",
         "v27-allrows-lidarshield", "v28-allrows-shield2", "v08b-ceiling-lookahead", "v29-lidarplan",
         "v31-lidarplan-alldata"]
# The verdict and why: the coordinator's reading of the numbers next to it (written by hand, numbers are not).
VERDICT = {
    "v03-reference": ("reference", "the dagger1 setup on v2 data: walks on where the expert stops and turns"),
    "v00-noise": ("noise floor", "v03 with another context seed: identical setups differ by about 4 points"),
    "v01-rule": ("baseline", "heading alone (stop when the goal is off to the side): crashes into what is ahead"),
    "v06-knn": ("baseline", "kNN on v03's rows and columns: TabPFN earns its place"),
    "v02-summary": ("better than v03", "8 columns beat 606: the patch grid drowns the goal columns"),
    "v09-scan": ("not better than v02", "a camera horizon scan instead of the summary: no gain"),
    "v10-band": ("not better", "camera horizon depth + DINOv2 PCA: the camera cannot see beside the robot"),
    "v11-lidar": ("promoted (1)", "the 360 degree lidar: most crashes were into furniture beside the robot"),
    "v12-lidar-perturb": ("not better", "heading-recovery rows: better offline stop recall, same driving"),
    "v13-lidar-gated": ("not better", "walk / stop-turn classifier: offline stop recall 51% -> 82%, same driving"),
    "v14-lidar-only": ("not better", "lidar without camera features: the camera adds little next to the lidar"),
    "v15-lidar-allrows": ("leader (2nd)", "all 31k rows instead of 8k: more real rows help"),
    "v16-lidar-history": ("killed", "short memory of the last 2 steps: no gain"),
    "v17-lidar-balanced": ("killed", "context balanced over mode and free space: no gain"),
    "v18-lidar-dagger1": ("not better", "DAgger round 1 rows in a 12k context: no gain"),
    "v19-lidar-dagger1-near": ("killed", "keeping every close-to-furniture row: hesitant, 5% stuck"),
    "v20-lidar-allrows-dagger1": ("not better", "all rows + DAgger round 1 (52k): plateau"),
    "v21-lidar-allrows-dagger12": ("not better", "all rows + DAgger rounds 1-2 (61k): plateau"),
    "v22c-collector": ("data collector", "12k student used to drive DAgger rounds 3-4: plateau"),
    "v23-gated-allrows": ("not better", "gated output with all rows: stops ignored 230 -> 130, crashes unchanged"),
    "v24-allrows-est8": ("not better", "8 estimators instead of 4: capacity is not the limit"),
    "v27-allrows-lidarshield": ("killed", "safety stop at 0.27 m: decides too late, crashes while turning"),
    "v28-allrows-shield2": ("killed", "earlier safety stop: trades crashes for stuck and spinning"),
    "v08b-ceiling-lookahead": ("ceiling (privileged)", "the expert's look-ahead bearing: the missing piece is path info"),
    "v29-lidarplan": ("promoted (2): best", "the robot's own plan on its lidar map: most of the ceiling"),
    "v31-lidarplan-alldata": ("not better", "v29 with every row (81k, incl. DAgger): more mixed data was worse"),
}


ROUND_STUDENT = {1: "v11-lidar", 2: "v18-lidar-dagger1", 3: "v22c-collector", 4: "v26c-collector"}


def _load(exp_id):
    d = experiment_dir(exp_id)
    if not (d / "rows.parquet").exists() or not (d / "episodes.json").exists():
        from .merge import merge_dir
        merge_dir(d, plan=None, log=lambda *a: None)  # strict checks: duplicates, gaps, rows vs episode summaries
    rows = pd.read_parquet(d / "rows.parquet")
    episodes = json.loads((d / "episodes.json").read_text())
    jobs = [json.loads(p.read_text()) for p in sorted(d.glob("job-*.json"))]
    fit = [r.get("info", {}) for j in jobs for r in j.get("results", []) if r.get("info")]
    info = fit[0] if fit else {}
    return rows, episodes, info


def _subset(rows, episodes, plan):
    keep = {e["episode"] for e in plan}
    eps = [e for e in episodes if e["episode"] in keep]
    if len(eps) != len(keep):
        raise ValueError("{} of {} planned episodes present".format(len(eps), len(keep)))
    return rows[rows["episode"].isin(keep)].reset_index(drop=True), eps


def student_only(rows):
    """True when the student drove every step: the expert's command never executed, executed == student."""
    if rows["used_expert"].any():
        return False
    return bool(np.allclose(rows["executed_stride"], rows["student_stride"].clip(0, 0.3), atol=1e-6)
                and np.allclose(rows["executed_turn"], rows["student_turn"].clip(-0.25, 0.25), atol=1e-6))


def _rate(s, key, digits=0):
    r = s[key]
    f = "{:." + str(digits) + "%}"
    return (f + " (" + f + "-" + f + ")").format(r["rate"], r["ci_low"], r["ci_high"]) if key == "goal" else f.format(r["rate"])


def screening_table(n=100):
    plan = set_plan("screen", 0, n)
    expert_rows = pd.read_parquet(dataset_dir("screen") / "rows.parquet")
    expert_eps = json.loads((dataset_dir("screen") / "episodes.json").read_text())
    expert_rows, expert_eps = _subset(expert_rows, expert_eps, plan)
    tables, out = {}, []
    for vid in ORDER:
        rows, eps, info = _load(vid)
        rows, eps = _subset(rows, eps, plan)
        tables[vid] = episode_table(rows, eps, expert_episodes=expert_eps)
    expert = summarize(episode_table(expert_rows, expert_eps))
    for vid in ORDER:
        _, _, info = _load(vid)
        s = summarize(tables[vid])
        ref = REFERENCE.get(vid)
        vs = compare(tables[vid], tables[ref]) if ref else None
        out.append({"id": vid, "verdict": VERDICT[vid][0], "why": VERDICT[vid][1], "reference": ref,
                    "columns": info.get("columns_fit"), "context_rows": info.get("context_rows"),
                    "fit_s": info.get("fit_seconds"), "student": info.get("student"), "summary": s, "vs": vs,
                    "student_only": student_only(_load(vid)[0])})
    return out, expert


def promotion_table():
    runs = [("screen", 200, ["v03-reference", "v11-lidar", "v15-lidar-allrows", "v29-lidarplan"]),
            ("confirm", 200, ["v29-lidarplan-confirm"]),
            ("final", 200, ["v29-lidarplan-final"]), ("final_b", 200, ["v29-lidarplan-final_b"])]
    out = []
    for set_name, n, ids in runs:
        plan = set_plan(set_name, 0, n)
        er = pd.read_parquet(dataset_dir(set_name) / "rows.parquet")
        ee = json.loads((dataset_dir(set_name) / "episodes.json").read_text())
        er, ee = _subset(er, ee, plan)
        out.append({"set": set_name, "id": "expert", "summary": summarize(episode_table(er, ee))})
        for vid in ids:
            rows, eps, _ = _load(vid)
            rows, eps = _subset(rows, eps, plan)
            out.append({"set": set_name, "id": vid, "summary": summarize(episode_table(rows, eps, expert_episodes=ee)),
                        "student_only": student_only(rows)})
    return out


def dagger_rounds():
    import yaml
    out = []
    for k in range(1, 5):
        d = dataset_dir("dagger_r{}".format(k))
        merge = json.loads((d / "merge.json").read_text()) if (d / "merge.json").exists() else {}
        variant = d / "student-variant.yaml"
        # the round's student; the variant file was written on metacon, where the rounds ran
        student = (yaml.safe_load(variant.read_text())["id"] if variant.exists()
                   else ROUND_STUDENT.get(k, "?"))
        rows = pd.read_parquet(d / "rows.parquet", columns=["used_expert", "beta"])
        out.append({"round": k, "student": student, "beta": float(rows["beta"].iloc[0]), "episodes": merge.get("episodes"),
                    "rows": merge.get("rows"), "student_rows": int((~rows["used_expert"]).sum()),
                    "outcomes": merge.get("outcomes")})
    return out


def _fmt_p(p):
    return "-" if p is None else ("<0.001" if p < 0.001 else "{:.3f}".format(p))


def write_markdown(folder, screen, expert, promo, rounds):
    lines = ["# Every variant tried (screening layouts 101000-101099, student alone, 150-step cap)", "",
             "Generated by `python -m nav.dagger.report` from `data/nav/dagger/v2/experiments/`. Each variant changes one "
             "thing against its reference; the comparison is paired on the same 100 layouts and start poses "
             "(wins = goal where the reference did not). The expert on these layouts: goal {:.0%}.".format(
                 expert["goal"]["rate"]), "",
             "Fit seconds are on the machine that ran the variant: vendetta (M4 Pro, MPS) for all but `v22c-collector` "
             "(metacon, CPU).", "",
             "| variant | verdict | the one change, and what it showed | cols | context rows | fit s | goal (95% CI) "
             "| collisions | stuck | vs reference: wins/losses, sign p, Wilcoxon p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in screen:
        s, vs = r["summary"], r["vs"]
        cmp_ = "-" if not vs else "{} {}/{} p={} W={}".format(r["reference"].split("-")[0], vs["wins"], vs["losses"],
                                                               _fmt_p(vs["sign_p"]), _fmt_p(vs["wilcoxon_p"]))
        lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["id"], r["verdict"], r["why"], r["columns"] or "-", "{:,}".format(r["context_rows"]) if r["context_rows"] else "-",
            "{:.0f}".format(r["fit_s"]) if r["fit_s"] is not None else "-", _rate(s, "goal"), _rate(s, "collision"),
            s["stuck"]["count"], cmp_))
    (folder / "experiments.md").write_text("\n".join(lines) + "\n")

    lines = ["# Promotion and robust checks (student alone)", "",
             "| set | layouts | who drives | goal (95% CI) | collisions (95% CI) | stuck | timeout |", "|---|---|---|---|---|---|---|"]
    for r in promo:
        s = r["summary"]
        c = s["collision"]
        lines.append("| {} | {} | {} | {} | {:.1%} ({:.1%}-{:.1%}) | {} | {} |".format(
            r["set"], s["n"], "expert (anchor)" if r["id"] == "expert" else "student `{}`".format(r["id"]),
            _rate(s, "goal", 1), c["rate"], c["ci_low"], c["ci_high"], s["stuck"]["count"], s["timeout"]["count"]))
    (folder / "promotion.md").write_text("\n".join(lines) + "\n")

    lines = ["# DAgger rounds (student drives on its own new training layouts, the expert labels every step)", "",
             "| round | student that drove | beta | episodes | outcomes | rows | student-driven rows |",
             "|---|---|---|---|---|---|---|"]
    for r in rounds:
        lines.append("| {} | `{}` | {} | {} | {} | {:,} | {:,} |".format(
            r["round"], r["student"], r["beta"], r["episodes"], ", ".join("{} {}".format(k, v) for k, v in
                                                                         sorted((r["outcomes"] or {}).items())),
            r["rows"] or 0, r["student_rows"]))
    (folder / "dagger_rounds.md").write_text("\n".join(lines) + "\n")


def results_chart(folder, promo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # One bar per model, split into what happened in the 200 runs: reached the target, hit furniture, neither.
    labels = {"v03-reference": "v03  camera only (where we started)",
              "v11-lidar": "v11  + 360° lidar (sees furniture beside it)",
              "v15-lidar-allrows": "v15  + all 31,586 expert examples",
              "v29-lidarplan": "v29  + plans a route on its own lidar map (best)",
              "expert": "the expert teacher (knows the true room)"}
    picks = [r for r in promo if r["set"] == "screen"]
    picks = [next(r for r in picks if r["id"] == k) for k in ("v03-reference", "v11-lidar", "v15-lidar-allrows",
                                                               "v29-lidarplan", "expert")][::-1]
    goal = np.array([r["summary"]["goal"]["rate"] * 100 for r in picks])
    coll = np.array([r["summary"]["collision"]["rate"] * 100 for r in picks])
    other = np.clip(100 - goal - coll, 0, None)
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=110)
    y = np.arange(len(picks))
    ax.barh(y, goal, color="#1f9d55", height=0.6, label="reached the target")
    ax.barh(y, coll, left=goal, color="#d64545", height=0.6, label="hit furniture")
    ax.barh(y, other, left=goal + coll, color="#b0b7bf", height=0.6, label="stuck or out of time")
    for i in range(len(picks)):
        ax.text(goal[i] - 1.5, i, "{:.1f}% reached".format(goal[i]), va="center", ha="right", color="white",
                fontsize=9, weight="bold")
        ax.text(101.5, i, "{:.1f}% crashed".format(coll[i]), va="center", ha="left", fontsize=9,
                color="#b02a2a" if coll[i] > 0 else "#555")
    ax.set_yticks(y, [labels[r["id"]] for r in picks], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of 200 rooms it had never seen (the model drives alone)")
    ax.set_title("How each model ended its 200 test runs (each row adds one thing to the row above)", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / "results.png")
    plt.close(fig)


def route_picture(folder, exp_id, set_name, episode, title, extra=None):
    """Student path (the trained model alone) vs the expert's path on the same layout and start pose."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .env import SimEnv
    from scene import HALF
    rows, _, _ = _load(exp_id)
    g = rows[rows["episode"] == episode].sort_values("step")
    if g.empty or not student_only(g):
        raise ValueError("episode {} of {} is not a student-only run".format(episode, exp_id))
    er = pd.read_parquet(dataset_dir(set_name) / "rows.parquet", columns=["episode", "step", "x", "y"])
    e = er[er["episode"] == episode].sort_values("step")
    config = load_config()
    env_cfg = config["env"]
    env = SimEnv(simple=env_cfg["simple"], target_clearance=env_cfg.get("target_clearance"),
                 target_wall=env_cfg.get("target_wall"))
    env.reset(int(g["scene_id"].iloc[0]), int(g["seed"].iloc[0]))
    feet = env.footprints()
    goal = env.goal
    env.close()
    panels = [(exp_id, g, title)] + ([extra] if extra else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), dpi=100, squeeze=False)
    for ax, (pid, path, ptitle) in zip(axes[0], panels):
        ax.set_xlim(-HALF - 0.1, HALF + 0.1)
        ax.set_ylim(-HALF - 0.1, HALF + 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(plt.Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False, lw=2, color="0.3"))
        for f in feet:
            if f["kind"] == "circle":
                ax.add_patch(plt.Circle((f["x"], f["y"]), f["r"], color="#d08a2a"))
            elif max(abs(f["x"]), abs(f["y"])) < HALF:
                c, s = math.cos(f["yaw"]), math.sin(f["yaw"])
                corner = (f["x"] - c * f["hx"] + s * f["hy"], f["y"] - s * f["hx"] - c * f["hy"])
                ax.add_patch(plt.Rectangle(corner, 2 * f["hx"], 2 * f["hy"], angle=math.degrees(f["yaw"]), color="#d08a2a"))
        ax.plot(e["x"], e["y"], color="#c9a227", lw=2, ls="--", label="expert (teacher, for comparison)")
        reason = path["done_reason"].iloc[-1]
        ax.plot(path["x"], path["y"], color="#1f9d55" if reason == "goal" else "#d33", lw=2.4,
                label="student alone: {}".format(reason))
        ax.plot(path["x"].iloc[0], path["y"].iloc[0], "o", color="#123a80", ms=7, label="start")
        ax.plot(*goal, "*", color="#e03030", ms=14, label="target")
        ax.set_title(ptitle, fontsize=9)
        ax.legend(loc="lower left", fontsize=6.5, framealpha=0.85)
    fig.tight_layout()
    out = folder / "route-{}.png".format(episode)
    fig.savefig(out)
    plt.close(fig)
    return out


KIND_NAME = {"collect": "expert episodes (data and reference sets)", "dagger": "DAgger round (student drives)",
             "run": "variant test (fit + student drives)"}


def job_records(roots):
    """Every job-<host>-<time>.json under the given folders, one record each.

    Each machine writes such a record at the end of every job (collection, DAgger round, variant test). The same
    file copied to another machine is counted once (keyed by its path below the root and its name).
    """
    seen = {}
    for root in roots:
        root = Path(root)
        for path in sorted(root.rglob("job-*.json")):
            key = str(path.relative_to(root))
            if key in seen:
                continue
            d = json.loads(path.read_text())
            results = [r for r in d.get("results") or [] if isinstance(r, dict)]
            fit = sum(float((r.get("info") or {}).get("fit_seconds") or 0) for r in results)
            seen[key] = {"host": d.get("host"), "kind": d.get("kind"), "label": d.get("label"),
                         "episodes": int(d.get("ran") or 0), "procs": int(d.get("procs") or 1),
                         "seconds": float(d.get("seconds") or 0), "fit_seconds": fit,
                         "steps": sum(int(r.get("steps") or 0) for r in results),
                         "finished": path.stat().st_mtime, "file": key}
    return [r for r in seen.values() if r["episodes"] > 0]


def compute_summary(records):
    """Per machine and per kind of job: jobs, busy hours, episodes, simulated steps (gait cycles), TabPFN fit hours."""
    out = {}
    for r in records:
        for key in ((r["host"], "all"), (r["host"], r["kind"]), ("all", "all"), ("all", r["kind"])):
            s = out.setdefault(key, {"jobs": 0, "hours": 0.0, "episodes": 0, "steps": 0, "fit_hours": 0.0})
            s["jobs"] += 1
            s["hours"] += r["seconds"] / 3600
            s["episodes"] += r["episodes"]
            s["steps"] += r["steps"]
            s["fit_hours"] += r["fit_seconds"] / 3600
    return out


def compute_markdown(records, dagger1=None):
    """compute.md: where the compute went, from the job records (and dagger1's timings.json if given)."""
    summary = compute_summary(records)
    hosts = sorted({r["host"] for r in records})
    lines = ["# Compute: which machine did what (DAgger v2, 2026-11-22)", "",
             "Generated by `python -m nav.dagger.report --compute-only` from the `job-<machine>-<time>.json` record "
             "each machine writes at the end of every job. Hours are wall-clock time of the job. One simulated step "
             "is one gait cycle (one second of robot time) and one row of data.", "",
             "| machine | kind of job | jobs | busy hours | episodes | simulated steps | TabPFN fit hours |",
             "|---|---|---|---|---|---|---|"]
    for host in hosts + ["all"]:
        for kind in ["collect", "dagger", "run", "all"]:
            s = summary.get((host, kind))
            if not s:
                continue
            name = "**all**" if kind == "all" else KIND_NAME[kind]
            lines.append("| {} | {} | {} | {:.2f} | {:,} | {:,} | {:.2f} |".format(
                "**{}**".format(host) if kind == "all" else host, name, s["jobs"], s["hours"], s["episodes"],
                s["steps"], s["fit_hours"]))
    lines += ["", "## Every job", "",
              "Finish times are the job record's file time, in the time zone of the machine that ran this report.", "",
              "| finished | machine | kind | what | episodes | processes | minutes | of which TabPFN fit | steps |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        lines.append("| {} | {} | {} | `{}` | {} | {} | {:.1f} | {} | {:,} |".format(
            datetime.fromtimestamp(r["finished"]).strftime("%d %b %H:%M"), r["host"], r["kind"], r["label"], r["episodes"], r["procs"], r["seconds"] / 60,
            "{:.1f}".format(r["fit_seconds"] / 60) if r["fit_seconds"] else "-", r["steps"]))
    if dagger1:
        lines += ["", "## dagger1 (the first loop, 2026-11-21), round 0 expert episodes", "",
                  "| machine | processes | episodes | steps | minutes | steps per second |", "|---|---|---|---|---|---|"]
        for t in dagger1:
            lines.append("| {} | {} | {} | {:,} | {:.1f} | {:.1f} |".format(
                t["host"], t["procs"], t["episodes"], t["steps"], t["rollout_s"] / 60, t["steps_per_s"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(v2_dir() / "report"))
    parser.add_argument("--jobs", nargs="*", default=None,
                        help="folders to search for job-*.json records (one per machine's data/nav/dagger/v2)")
    parser.add_argument("--dagger1-timings", default=None, help="dagger1's timings.json")
    parser.add_argument("--compute-only", action="store_true", help="only write compute.md")
    args = parser.parse_args()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    if args.jobs is not None:
        records = sorted(job_records(args.jobs), key=lambda r: r["finished"])
        dagger1 = json.loads(Path(args.dagger1_timings).read_text()) if args.dagger1_timings else None
        (folder / "compute.md").write_text(compute_markdown(records, dagger1))
        print("written", folder / "compute.md")
    if args.compute_only:
        return
    screen, expert = screening_table()
    promo = promotion_table()
    rounds = dagger_rounds()
    write_markdown(folder, screen, expert, promo, rounds)
    results_chart(folder, promo)
    (folder / "report_data.json").write_text(json.dumps({"screening": screen, "expert_screen": expert,
                                                          "promotion": promo, "dagger_rounds": rounds}, indent=1,
                                                         default=str))
    for exp_id, set_name, ep, title in (
            ("v29-lidarplan-final_b", "final_b", 105061, "final test 105061: student alone (goal)"),
            ("v29-lidarplan-final", "final", 102155, "final test 102155: student alone (goal)"),
            ("v29-lidarplan-final", "final", 102093, "final test 102093: student alone (goal)"),
            ("v29-lidarplan-final", "final", 102027, "final test 102027: student alone (one of 4 crashes)")):
        print(route_picture(folder, exp_id, set_name, ep, title))
    # before / after: the same screening layout, camera-grid reference vs the best model
    ref_rows, ref_eps, _ = _load("v03-reference")
    best_rows, best_eps, _ = _load("v29-lidarplan")
    ref_bad = {e["episode"] for e in ref_eps if e["done_reason"] == "collision"}
    best_good = {e["episode"] for e in best_eps if e["done_reason"] == "goal"}
    both = sorted(ref_bad & best_good)
    if both:
        ep = both[0]
        ref_path = ref_rows[ref_rows["episode"] == ep].sort_values("step")
        print(route_picture(folder, "v29-lidarplan", "screen", ep, "screening {}: best model v29 (goal)".format(ep),
                            extra=("v03", ref_path, "same layout: camera-grid reference v03 (crash)")))
    print("student-only checks:", {r["id"]: r["student_only"] for r in screen})
    print("written to", folder)


if __name__ == "__main__":
    main()
