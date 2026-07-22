"""ParaHand FR3 robot configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

PARAHAND_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "parahand_fr3" / "xmls" / "parahand_fr3.xml"
)
assert PARAHAND_XML.exists()

ARM_ACTUATOR_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")
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


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(PARAHAND_XML))


PARAHAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(
      target_names_expr=ARM_ACTUATOR_NAMES + HAND_ACTUATOR_NAMES,
      command_field="position",
    ),
    XmlActuatorCfg(
      target_names_expr=TENDON_ACTUATOR_NAMES,
      transmission_type=TransmissionType.TENDON,
      command_field="position",
    ),
  ),
)

PARAHAND_INITIAL_STATE = EntityCfg.InitialStateCfg(
  joint_pos=None,
  joint_vel={".*": 0.0},
)


def get_parahand_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    articulation=PARAHAND_ARTICULATION,
    init_state=PARAHAND_INITIAL_STATE,
  )


PARAHAND_ACTION_SCALE = {
  **dict.fromkeys(ARM_ACTUATOR_NAMES, 0.01),
  **dict.fromkeys(HAND_ACTUATOR_NAMES, 0.03),
  **dict.fromkeys(TENDON_ACTUATOR_NAMES, 0.01),
}
