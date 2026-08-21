"""Free cube asset configuration.

Each cube entity receives randomized physical and visual properties
when its MuJoCo specification is created:

- size
- mass
- color

The properties remain fixed for that environment/model and the
cube position and activation state are randomized at every reset.
"""

from pathlib import Path
import random

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

# =========================================================
# VIVID COLORS
# =========================================================
#
# RGB values are in the [0, 1] range.
#

CUBE_COLORS = (
    (0.05, 0.35, 1.00, 1.0),  # blue
    (1.00, 0.05, 0.05, 1.0),  # red
    (0.05, 0.85, 0.15, 1.0),  # green
    (1.00, 0.80, 0.00, 1.0),  # yellow
    (0.65, 0.10, 1.00, 1.0),  # purple
    (1.00, 0.35, 0.02, 1.0),  # orange
)


def _randomize_cube_spec(
    spec: mujoco.MjSpec,
) -> mujoco.MjSpec:
    """Randomize the physical and visual properties of one cube."""

    # ---------------------------------------------------------
    # Random cube size.
    # ---------------------------------------------------------

    half_size = random.uniform(
        CUBE_HALF_SIZE_MIN,
        CUBE_HALF_SIZE_MAX,
    )

    # ---------------------------------------------------------
    # Random cube mass.
    # ---------------------------------------------------------

    mass = random.uniform(
        CUBE_MASS_MIN,
        CUBE_MASS_MAX,
    )

    # ---------------------------------------------------------
    # Random vivid color.
    # ---------------------------------------------------------

    rgba = random.choice(CUBE_COLORS)

    # ---------------------------------------------------------
    # Apply geometry properties.
    # ---------------------------------------------------------

    if len(spec.geoms) != 1:
        raise ValueError(
            "Expected exactly one geom in cube.xml, "
            f"found {len(spec.geoms)}."
        )

    geom = spec.geoms[0]

    # MuJoCo box size = half-extents.
    geom.size = (
        half_size,
        half_size,
        half_size,
    )

    geom.mass = mass

    # ---------------------------------------------------------
    # Apply material color.
    # ---------------------------------------------------------

    if len(spec.materials) != 1:
        raise ValueError(
            "Expected exactly one material in cube.xml, "
            f"found {len(spec.materials)}."
        )

    material = spec.materials[0]

    material.rgba = rgba

    return spec


def get_spec() -> mujoco.MjSpec:
    """Load and randomize the MuJoCo cube specification."""

    spec = mujoco.MjSpec.from_file(
        str(CUBE_XML)
    )

    return _randomize_cube_spec(spec)


def get_cube_cfg() -> EntityCfg:
    """Return a cube entity configuration with randomized properties."""

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        spec_fn=get_spec,
    )