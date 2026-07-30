uv run train Mjlab-Grasp-Object-ParaHand-Only \
  --gpu-ids "[0, 1, 2]" \
  --env.scene.num-envs 1365 \
  --agent.max-iterations 30000 \
  --agent.save-interval 300 \
  --agent.logger wandb \
  --agent.wandb-project parahand \
  --agent.stage2-dataset robustdex