from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class PrimitiveObject:
  name: str
  geom_type: mujoco.mjtGeom
  size: tuple[float, float, float]
  geom_quat: tuple[float, float, float, float]
  rgba: tuple[float, float, float, float]
  floor_offset: float


PRIMITIVE_OBJECTS = (
  PrimitiveObject(
    name="object_box",
    geom_type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.03, 0.03, 0.03),
    geom_quat=(1.0, 0.0, 0.0, 0.0),
    rgba=(0.0, 1.0, 0.0, 1.0),
    floor_offset=0.03,
  ),
  PrimitiveObject(
    name="object_sphere",
    geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(0.03, 0.0, 0.0),
    geom_quat=(1.0, 0.0, 0.0, 0.0),
    rgba=(0.1, 0.4, 1.0, 1.0),
    floor_offset=0.03,
  ),
  PrimitiveObject(
    name="object_capsule",
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=(0.02, 0.035, 0.0),
    geom_quat=(0.7071067811865476, -0.7071067811865475, 0.0, 0.0),
    rgba=(0.7, 0.2, 1.0, 1.0),
    floor_offset=0.02,
  ),
)

PRIMITIVE_OBJECT_NAMES = tuple(obj.name for obj in PRIMITIVE_OBJECTS)
