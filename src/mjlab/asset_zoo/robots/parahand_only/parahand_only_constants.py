"""ParaHand-only robot configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

PARAHAND_ONLY_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "parahand_only" / "xmls" / "hand_only.xml"
)
assert PARAHAND_ONLY_XML.exists()

PALM_TRANSLATION_ACTUATOR_NAMES = (
  "palm_translation_x",
  "palm_translation_y",
  "palm_translation_z",
)
PALM_ROTATION_ACTUATOR_NAMES = (
  "palm_rotation_x",
  "palm_rotation_y",
  "palm_rotation_z",
)
HAND_ACTUATOR_NAMES = (
  "thumb_cmc_1",
  "thumb_cmc_2",
  "thumb_mcp",
  "thumb_ip",
  "index_mcp_1",
  "index_mcp_2",
  "middle_mcp_1",
  "middle_mcp_2",
  "ring_mcp_1",
  "ring_mcp_2",
  "little_mcp_1",
  "little_mcp_2",
)
TENDON_ACTUATOR_NAMES = (
  "index_tendon",
  "middle_tendon",
  "ring_tendon",
  "little_tendon",
)
JOINT_ACTUATOR_NAMES = (
  PALM_TRANSLATION_ACTUATOR_NAMES + PALM_ROTATION_ACTUATOR_NAMES + HAND_ACTUATOR_NAMES
)


def get_spec() -> mujoco.MjSpec:
  """Load the hand asset without its standalone demo cube and floor."""
  spec = mujoco.MjSpec.from_file(str(PARAHAND_ONLY_XML))
  if len(spec.keys) != 1 or spec.keys[0].name != "home":
    raise ValueError("ParaHand-only XML must define exactly one 'home' keyframe.")
  if spec.joints[-1].name != "cube_freejoint":
    raise ValueError("ParaHand-only demo cube freejoint must be the final joint.")

  xml_home = spec.keys[0]
  home_qpos = list(xml_home.qpos)[:-7]
  home_qvel = list(xml_home.qvel)[:-6]
  home_ctrl = list(xml_home.ctrl)
  spec.delete(xml_home)
  spec.delete(spec.body("cube"))
  spec.delete(spec.geom("floor"))

  spec.add_key(
    name="home",
    qpos=home_qpos,
    qvel=home_qvel,
    ctrl=home_ctrl,
  )
  return spec


PARAHAND_ONLY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(
      target_names_expr=JOINT_ACTUATOR_NAMES,
      command_field="position",
    ),
    XmlActuatorCfg(
      target_names_expr=TENDON_ACTUATOR_NAMES,
      transmission_type=TransmissionType.TENDON,
      command_field="position",
    ),
  ),
)

PARAHAND_ONLY_INITIAL_STATE = EntityCfg.InitialStateCfg(
  joint_pos=None,
  joint_vel={".*": 0.0},
)


def get_parahand_only_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    articulation=PARAHAND_ONLY_ARTICULATION,
    init_state=PARAHAND_ONLY_INITIAL_STATE,
  )


PARAHAND_ONLY_ACTION_SCALE = {
  **dict.fromkeys(PALM_TRANSLATION_ACTUATOR_NAMES, 0.01),
  **dict.fromkeys(PALM_ROTATION_ACTUATOR_NAMES, 0.02),
  **dict.fromkeys(HAND_ACTUATOR_NAMES, 0.05),
  **dict.fromkeys(TENDON_ACTUATOR_NAMES, 0.005),
}
