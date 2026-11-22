"""Pictures of Margarete in the app's own flat, for her DINOv2 signature: just person-data [--flats 48]

The detector (nav/dagger/target.py) matches the camera's horizon band (DINOv2 features of patch rows 11-14,
averaged per patch column: 46 columns x 384) to a signature. The pillar's signature came from the plain arena;
Margarete lives in the dressed flat (wood floor, zone patches, decor, furniture in its home colours), so her
signature is made here, from the scene the app renders (`ApartmentScene`):

  - random flats (homes/flats.py), half of them with the loose pieces scattered;
  - zone patches painted with random beliefs, as during a search;
  - her at a random spot of a random zone, lying or standing (the app's two poses), or nobody home (20%);
  - the robot at a free place: mostly looking at her from 0.5-4 m (yaw jitter 0.45 rad), else anywhere.

Per picture: the band (46 x 384, float16) and the share of her pixels in every band column (from MuJoCo's
segmentation, geoms named person_*). Written to data/vision/<run>/ (gitignored): band.npy, share.npy, meta.json.
Then `python vision/target_signature.py <run> --band` builds signature-margarete.npz.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent
REPO = APP.parent
for p in (APP, REPO, REPO / "vision"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

BAND_ROWS = (11, 15)  # patch rows, as nav/dagger/featurize.py
AIMED = 0.75          # share of pictures aimed at her
DIST = (0.5, 4.0)     # metres from her, aimed pictures
JITTER = 0.45         # radians of yaw around the bearing to her
NOBODY = 0.2          # share of pictures with nobody home


def band_share(seg_person):
    """(360, 640) bool, her pixels -> (46,) share of her pixels in each band column, on DINOv2's 26 x 46 grid."""
    h, w = seg_person.shape
    rows = np.arange(26 * 14) * h // (26 * 14)
    cols = np.arange(46 * 14) * w // (46 * 14)
    big = seg_person[rows][:, cols]  # (364, 644), as features.extract resizes
    band = big[BAND_ROWS[0] * 14:BAND_ROWS[1] * 14]
    return band.reshape(band.shape[0], 46, 14).mean(axis=(0, 2)).astype(np.float32)


def free(scene, x, y, half):
    """The robot standing at (x, y) touches no wall or furniture (and stays 0.3 m off the walls)."""
    if abs(x) > half - 0.3 or abs(y) > half - 0.3:
        return False
    import mujoco
    sim = scene.sim
    sim.data.qpos[0], sim.data.qpos[1] = x, y
    mujoco.mj_forward(sim.model, sim.data)
    model, data = sim.model, sim.data
    for i in range(data.ncon):
        names = (model.geom(int(data.contact.geom1[i])).name, model.geom(int(data.contact.geom2[i])).name)
        if any(n.startswith(("wall_", "obs_")) for n in names) and any(not n.startswith(("wall_", "obs_", "floor"))
                                                                        for n in names):
            return False
    return True


def main():
    import mujoco  # noqa: F401
    from features import extract, load_dino
    from dataset import data_dir
    from homes import flats as F
    from robot import ApartmentScene

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--flats", type=int, default=48)
    parser.add_argument("--pictures", type=int, default=24, help="per flat")
    parser.add_argument("--first-flat", type=int, default=5000, help="flat ids (seeds) start here")
    parser.add_argument("--name", default="person-v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = data_dir() / args.name
    out.mkdir(parents=True, exist_ok=True)
    scene = ApartmentScene()
    model = scene.sim.model
    person_geoms = np.array([model.geom(g).name.startswith("person_") for g in range(model.ngeom)] + [False])
    dino = load_dino()
    rng = np.random.default_rng(args.seed)
    bands, shares, meta = [], [], []
    started = time.time()
    for k in range(args.flats):
        flat_id = args.first_flat + k
        flat = F.random_flat(np.random.default_rng(flat_id), flat_id)
        scattered = bool(rng.random() < 0.5)
        if scattered:
            flat = F.scatter(flat, np.random.default_rng([flat_id, 1]))
        scene.load(flat)
        half = scene.half
        pictures = []
        for _ in range(args.pictures):
            scene.paint({z["name"]: float(p) for z, p in zip(flat["zones"], rng.dirichlet(np.ones(len(flat["zones"]))))})
            nobody = rng.random() < NOBODY
            zone = flat["zones"][int(rng.integers(len(flat["zones"])))]
            spot = zone["spots"][int(rng.integers(len(zone["spots"])))]
            pose = "lying" if rng.random() < 0.5 else "standing"
            px, py, yaw = spot["person_x"], spot["person_y"], math.radians(spot["heading_deg"] + 90.0)
            aimed = (not nobody) and rng.random() < AIMED
            for _ in range(200):
                if aimed:
                    d, a = rng.uniform(*DIST), rng.uniform(-math.pi, math.pi)
                    x, y = px + d * math.cos(a), py + d * math.sin(a)
                    ryaw = math.atan2(py - y, px - x) + rng.uniform(-JITTER, JITTER)
                else:
                    x, y = rng.uniform(-half + 0.3, half - 0.3, size=2)
                    ryaw = rng.uniform(-math.pi, math.pi)
                scene.place((float(x), float(y), float(ryaw)))
                if free(scene, float(x), float(y), half):
                    break
            else:
                continue
            scene.place((float(x), float(y), float(ryaw)))
            if nobody:
                scene.person()
            else:
                scene.person(px, py, yaw, pose)
            mujoco.mj_forward(scene.sim.model, scene.sim.data)
            rgb = scene.sim.render()
            seg = scene.sim.render(segmentation=True)
            share = band_share(person_geoms[seg])
            pictures.append(rgb)
            shares.append(share)
            meta.append({"flat": flat_id, "scattered": scattered, "zone": zone["name"],
                         "pose": "none" if nobody else pose, "person": [px, py], "robot": [x, y, ryaw],
                         "dist": None if nobody else math.hypot(px - x, py - y), "aimed": aimed})
        for i in range(0, len(pictures), 8):
            feats = extract(dino, np.stack(pictures[i:i + 8]))
            bands.append(feats[:, BAND_ROWS[0]:BAND_ROWS[1]].mean(axis=1).astype(np.float16))
        seen = sum(1 for m, s in zip(meta[-len(pictures):], shares[-len(pictures):]) if s.max() > 0)
        print("flat {} ({}): {} pictures, she is in view in {} ({:.0f} s)".format(
            flat_id, "scattered" if scattered else "home", len(pictures), seen, time.time() - started), flush=True)
    band = np.concatenate(bands)
    share = np.stack(shares)
    assert len(band) == len(share) == len(meta)
    np.save(out / "band.npy", band)
    np.save(out / "share.npy", share)
    (out / "meta.json").write_text(json.dumps({"flats": args.flats, "first_flat": args.first_flat,
                                               "band_rows": list(BAND_ROWS), "frames": meta}))
    print("saved {} pictures to {}".format(len(meta), out))


if __name__ == "__main__":
    main()
