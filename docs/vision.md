# Vision: which camera patches are wall, obstacle, target or floor

For obstacle avoidance the robot needs to know, for every part of its camera picture, what it is looking at.
`vision/` answers that per **patch**: the frozen DINOv2 model (`dinov2_vits14_reg`) cuts the 640 x 360 picture
into 14 x 14 pixel patches and describes each with 384 numbers, and a small **probe** (one linear layer, or one
hidden layer of 256) turns those numbers into one of four classes: floor, wall, obstacle, target.

DINOv2's weights never change. Only the probe is trained, on labels that come for free from the simulator.

## Run it

```bash
just vision-data --episodes 60 --poses 16   # 960 labeled pictures in the arena, a few minutes
just vision-features latest                 # DINOv2 features and patch labels (downloads 88 MB once)
just vision-probe latest                    # train and test the linear probe
just vision-probe latest --hidden 256       # the same with one hidden layer
```

Everything is written to `data/vision/<run>/` in the main checkout (gitignored, shared by worktrees; set
`VISION_DATA_DIR` to put it elsewhere): `images/`, `labels.npz`, `meta.json`, `features.npy` (about 0.9 GB for
960 pictures), `patch_labels.npy`, `patch_counts.npy`, `probe.pt`, `metrics.json` and `overlays/` (held-out
pictures with the predicted classes painted on, and the blocked patches darkened).

## How the data is made

- **Scene**: the arena ([simulation.md](simulation.md#the-arena-a-closed-room-with-obstacles)), a closed 50 m²
  square room with 4.5 m walls and 8 obstacles.
- **Episode**: one random layout: obstacle positions, rotations and colors (never red), and the target.
- **Pictures**: 16 robot poses per episode on free floor, 70% of them looking roughly at the target from 0.4 to
  3 m. Every second picture is taken after a short random walk, so the body is tilted mid-step.
- **Pixel labels**: `sim.render(segmentation=True)` gives the geom at every pixel; its name prefix gives the
  class (`floor`, `wall_*`, `obs_*`, `target_geom`; the robot's own legs are 255, ignore).

## From pixel labels to patch labels

The picture is resized to 644 x 364 (multiples of 14): 26 rows x 46 columns = 1,196 patches. The class map is
resized the same way by nearest neighbour, so classes are never blended.

- A patch's label is its **majority class**, the class most of its 196 pixels have.
- **Purity** is the share of the patch's pixels that have the majority class. Floor, wall and obstacle patches
  under **70%** purity sit on an edge and are left out of training and of the four-class score
  (`--purity`).
- The **target** is thin (6 cm), so a patch is "target" when at least **30%** of its pixels are target
  (`--target-purity`).
- Patches that mostly show the robot's own legs are left out.

## Training and testing

- **Split by episode**: 20% of the layouts are held out, so the test shows furniture in places and colors the
  probe has not seen.
- Floor and wall patches are capped at 150 per picture; obstacle and target patches are all kept; the loss
  weights classes by inverse frequency.
- AdamW, learning rate 1e-3, weight decay 1e-4, 20 epochs; features are standardized with the training set's
  statistics, which are saved with the probe.

## For obstacle avoidance: the blocked grid

```python
import sys
sys.path.insert(0, "vision")
from probe import PatchClassifier

classifier = PatchClassifier("data/vision/<run>/probe.pt")
classes = classifier.predict_patches(rgb)    # (26, 46): 0 floor, 1 wall, 2 obstacle, 3 target
blocked = classifier.blocked_grid(rgb)       # (26, 46) True = stay away
```

`blocked_grid` leans toward caution: a patch is blocked when the probe gives wall + obstacle a probability over
**0.25** (not 0.5), and the blocked region is then grown by **one patch** in every direction. The second step
is what catches edge patches that are only partly obstacle: every such patch sits next to a pure obstacle patch.

`metrics.json` reports, on held-out episodes:

- **blocked vs free**: the share of wall or obstacle patches left free (the dangerous error) and the share of
  floor patches marked blocked (the cost of being careful), before and after the one-patch margin;
- **edge patches**: the skipped mixed patches, bucketed by how much of them is wall or obstacle, and how many
  are blocked;
- the four-class confusion matrix, recall, precision and balanced accuracy.

## Results

Run `full60` (2026-11-21): 960 pictures, 48 layouts for training, 12 held out (223,304 held-out patches).
Making the data and features took under 2 minutes on an Apple Silicon Mac (MPS); training takes seconds.

| | Linear probe | One hidden layer (256) |
|---|---|---|
| Accuracy / balanced accuracy | 98.8% / 98.7% | 99.3% / 99.0% |
| Recall floor, wall, obstacle, target | 99.2, 98.6, 98.4, 98.6% | 99.5, 99.3, 99.1, 98.1% |
| Target precision | 82.2% | 91.0% |
| Wall or obstacle left free (after the margin) | 0.41% (0.01%) | 0.18% (0.00%) |
| Floor marked blocked (after the margin) | 0.76% (12.5%) | 0.42% (12.1%) |
| Edge patches with 10 to 20% obstacle, blocked after the margin | 99.7% | 100% |

- The linear probe already separates the classes; the hidden layer mainly helps target precision (fewer
  wall and floor patches called target).
- The margin brings solid patches left free to about zero; its cost is that 12% of floor patches, the ones
  next to furniture and walls, are blocked too.

## Limits

- Simulation only: flat colors, even light, no camera noise. Expect a drop on real pictures.
- The probe sees one patch at a time. The camera's depth (`sim.render(depth=True)`) is a second, independent
  signal for "something is close" and is not used yet.
