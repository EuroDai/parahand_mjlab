from __future__ import annotations

from dataclasses import dataclass

import mujoco

FIRST_LESSON_OBJECT_NAME = "object_capsule"
PRIMITIVE_DATASET_STAGE = 6
PRIMITIVE_RANDOMIZATION_FRACTIONS = (0.0, 0.0, 0.1, 0.25, 0.5, 1.0)
CAPSULE_SCALE_RANGE = (0.5, 1.25)
BOX_SPHERE_SCALE_RANGE = (0.5, 1.0)
# Backward-compatible aggregate range for downstream imports.
OBJECT_SCALE_RANGE = BOX_SPHERE_SCALE_RANGE


def primitive_randomization_fraction(stage: int) -> float:
  if not 0 <= stage < PRIMITIVE_DATASET_STAGE:
    raise ValueError(
      f"Primitive curriculum stage must be in [0, {PRIMITIVE_DATASET_STAGE - 1}]."
    )
  return PRIMITIVE_RANDOMIZATION_FRACTIONS[stage]


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

PRIMITIVE_OBJECT_NAMES = tuple(obj.name for obj in PRIMITIVE_OBJECTS)
