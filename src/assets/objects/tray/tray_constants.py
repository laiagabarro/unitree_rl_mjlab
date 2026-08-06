"""Food tray asset configuration."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets


FOOD_TRAY_XML: Path = (
  SRC_PATH / "assets" / "objects" / "tray" / "xmls" / "food_tray.xml"
)
assert FOOD_TRAY_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, FOOD_TRAY_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(FOOD_TRAY_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_tray_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.32, 0.18, 0.98),
      quat=(1.0, 0.0, 0.0, 0.0),
    ),
    spec_fn=get_spec,
  )
