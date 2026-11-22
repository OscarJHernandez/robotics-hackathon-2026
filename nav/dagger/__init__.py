"""DAgger for a TabPFN navigation policy in the arena (see docs/navigation.md, "DAgger").

The student (TabPFN, `fit()` and `predict()` only) drives, the expert (a planner over the true obstacle
layout plus pure pursuit) labels every visited state, the rows are added to the data set and TabPFN is
fitted again on the bigger table. Run modules with `python -m nav.dagger.<module>` from the repo root.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (min, max) stride of the DAgger robot, radians of coxa swing. Negative = walking backwards (measured:
# -0.10 moves 4 cm back per 1 s cycle, -0.15 6 cm); the expert backs away from edges it is too close to.
# The rest of nav/ keeps nav.common.STRIDE_RANGE (0, 0.3).
STRIDE = (-0.15, 0.30)

# Headless rendering on Linux: MuJoCo reads MUJOCO_GL when it is first imported, so set it here.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

for _sub in ("sim", "vision"):
    _path = str(REPO / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
