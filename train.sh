uv run train Mjlab-Grasp-Object-ParaHand \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 15000 \
  --agent.save-interval 300 \
  --agent.logger wandb \
  --agent.wandb-project parahand