"""MJCF assembly for the fixed-lighting Phase 1 PickCube scene.

The SO-101 body, inertias, actuator limits, collision meshes, and gripper site
are vendored unchanged from TheRobotStudio/SO-ARM100 (Apache-2.0).  This module
only supplies the tabletop, free cube, and external camera.
"""
from __future__ import annotations

from pathlib import Path

ASSET_DIR = Path(__file__).with_name("assets")
ROBOT_XML = ASSET_DIR / "so101_new_calib.xml"


def scene_xml() -> str:
    """Return the complete task MJCF.

    World: right-handed MuJoCo world frame, +Z up. The tabletop's upper face is
    Z=0; the robot CAD base rests at that plane. Cube qpos is XYZ + WXYZ.
    """
    return """<mujoco model="mini_rdt_so101_pick_cube">
  <include file="assets/so101_new_calib.xml"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <size nconmax="400" njmax="1000"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="4096"/></visual>
  <asset>
    <texture name="table_tex" type="2d" builtin="flat" rgb1="0.42 0.28 0.16" width="32" height="32"/>
    <material name="table_mat" texture="table_tex" texuniform="true" rgba="1 1 1 1"/>
    <material name="cube_mat" rgba="0.82 0.08 0.05 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.1 -0.4 1.1" dir="0.1 0.3 -1" directional="true" diffuse="0.85 0.85 0.85" ambient="0.20 0.20 0.20"/>
    <light name="fill" pos="-0.5 0.3 0.7" dir="0.5 -0.3 -0.7" directional="true" diffuse="0.20 0.20 0.20"/>
    <geom name="table" type="box" pos="0.12 0 -0.04" size="0.48 0.42 0.04" material="table_mat" friction="1.2 0.01 0.001"/>
    <body name="cube" pos="0.23 0 0.02">
      <freejoint name="cube_freejoint"/>
      <geom name="cube_geom" type="box" size="0.01 0.01 0.01" mass="0.010" material="cube_mat" friction="1.8 0.02 0.001"/>
    </body>
    <camera name="external_rgb" pos="0.55 -0.62 0.52" xyaxes="0.74 0.67 0 -0.31 0.34 0.89" fovy="48"/>
  </worldbody>
</mujoco>"""


def write_scene(path: Path) -> Path:
    """Materialize the MJCF next to the asset directory for relative includes."""
    path.write_text(scene_xml(), encoding="utf-8")
    return path
