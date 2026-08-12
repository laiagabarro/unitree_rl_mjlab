"""Free cube asset configuration.

The cube is a lightweight rigid box meant to rest on top of the tray,
held in place by gravity + friction alone.

The cube is NOT welded to the tray.

The geometry uses MuJoCo box half-extents, so:

    CUBE_HALF_SIZE = 0.03 m

means:

    full cube size = 0.06 m
    = 6 cm
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


# MuJoCo `box size` is specified using half-extents.
#
# 0.03 m -> 6 cm full cube.
CUBE_HALF_SIZE: float = 0.03


def get_spec() -> mujoco.MjSpec:
    """Load the MuJoCo cube specification."""

    spec = mujoco.MjSpec.from_file(
        str(CUBE_XML)
    )

    return spec


def get_cube_cfg() -> EntityCfg:
    """Return the cube entity configuration."""

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        spec_fn=get_spec,
    )
