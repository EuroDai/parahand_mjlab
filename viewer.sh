_VISER_PORT_OVERRIDE=18701 \
uv run play Mjlab-Grasp-Object-ParaHand \
  --viewer viser \
  --device cuda:1 \
  --num-envs 12 \
  --checkpoint-file logs/rsl_rl/parahand_grasp_object/2026-07-22_16-26-09/model_0.pt