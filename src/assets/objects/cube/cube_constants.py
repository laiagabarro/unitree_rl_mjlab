"""Cube asset configuration.

The cube is created from a fixed MuJoCo specification.

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
# 0.02 -> 4 cm full cube
# 0.03 -> 6 cm full cube
# 0.04 -> 8 cm full cube

CUBE_HALF_SIZE_MIN: float = 0.02
CUBE_HALF_SIZE_MAX: float = 0.04


# Mass:
#
# 0.05 -> 50 g
# 0.15 -> 150 g (default)
# 0.30 -> 300 g

CUBE_MASS_MIN: float = 0.05
CUBE_MASS_MAX: float = 0.30


# Default mass defined in cube.xml.
CUBE_DEFAULT_MASS: float = 0.15


def get_spec() -> mujoco.MjSpec:
    """Load the fixed MuJoCo cube specification."""

    return mujoco.MjSpec.from_file(str(CUBE_XML))


def get_cube_cfg() -> EntityCfg:
    """Return the cube entity configuration."""

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        spec_fn=get_spec,
    )