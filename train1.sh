uv run train Mjlab-Grasp-Object-ParaHand-Only \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 2048 \
  --agent.resume True \
  --agent.load-run "2026-07-27_14-38-05" \
  --agent.load-checkpoint "model_3300.pt" \
  --agent.max-iterations 28500 \
  --agent.save-interval 300 \
  --agent.logger wandb \
  --agent.wandb-project parahand