from __future__ import annotations

from dataclasses import dataclass

import mujoco

FIRST_LESSON_OBJECT_NAME = "object_capsule"
OBJECT_SCALE_RANGE = (0.75, 1.5)
OBJECT_SCALE_FACTORS = (
  1.0,
  0.75,
  0.8125,
  0.875,
  0.9375,
  1.0625,
  1.125,
  1.1875,
  1.25,
  1.3125,
  1.375,
  1.4375,
  1.5,
)


@dataclass(frozen=True)
class PrimitiveObject:
  name: str
  geom_type: mujoco.mjtGeom
  size: tuple[float, float, float]
  geom_quat: tuple[float, float, float, float]
  rgba: tuple[float, float, float, float]
  floor_offset: float


_BASE_PRIMITIVE_OBJECTS = (
  PrimitiveObject(
    name=FIRST_LESSON_OBJECT_NAME,
    geom_type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=(0.02, 0.035, 0.0),
    geom_quat=(0.7071067811865476, -0.7071067811865475, 0.0, 0.0),
    rgba=(0.7, 0.2, 1.0, 1.0),
    floor_offset=0.02,
  ),
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
)


def _scaled_primitive(obj: PrimitiveObject, scale: float) -> PrimitiveObject:
  if scale == 1.0:
    return obj
  scale_suffix = f"{scale:.4f}".rstrip("0").rstrip(".").replace(".", "_")
  return PrimitiveObject(
    name=f"{obj.name}_scale_{scale_suffix}",
    geom_type=obj.geom_type,
    size=(
      obj.size[0] * scale,
      obj.size[1] * scale,
      obj.size[2] * scale,
    ),
    geom_quat=obj.geom_quat,
    rgba=obj.rgba,
    floor_offset=obj.floor_offset * scale,
  )


PRIMITIVE_OBJECTS = tuple(
  _scaled_primitive(obj, scale)
  for obj in _BASE_PRIMITIVE_OBJECTS
  for scale in OBJECT_SCALE_FACTORS
)

PRIMITIVE_OBJECT_NAMES = tuple(obj.name for obj in PRIMITIVE_OBJECTS)
