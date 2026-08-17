"""CUBE asset configuration (simplified as a box for now).

Its physical properties are randomized at every environment reset
through the domain-randomization events in the task:

- size
- mass + inertia

The color remains fixed.
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.entity import EntityCfg


CUBE_XML: Path = (
    SRC_PATH
    / "assets"
    / "objects"
    / "cube"
    / "xmls"
    / "cube.xml"
)

assert CUBE_XML.exists()


# =========================================================
# RANDOMIZATION RANGES
# =========================================================

# MuJoCo box `size` uses half-extents.
#
# 0.025 -> 5 cm full CUBE
# 0.035 -> 7 cm full CUBE (default)
# 0.045 -> 9 cm full CUBE

CUBE_HALF_SIZE_MIN: float = 0.025
CUBE_HALF_SIZE_MAX: float = 0.045


# Mass:
#
# 0.15 -> 150 g (empty mug)
# 0.30 -> 300 g (default)
# 0.40 -> 400 g (full mug)

CUBE_MASS_MIN: float = 0.15
CUBE_MASS_MAX: float = 0.40


# Default mass defined in CUBE.xml.
CUBE_DEFAULT_MASS: float = 0.30


def get_spec() -> mujoco.MjSpec:
    """Load the fixed MuJoCo CUBE specification."""

    return mujoco.MjSpec.from_file(str(CUBE_XML))


def get_cube_cfg() -> EntityCfg:
    """Return the CUBE entity configuration."""

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        spec_fn=get_spec,
    )