uv run train Mjlab-Grasp-Object-ParaHand \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1000 \
  --agent.save-interval 20 \
  --agent.logger wandb \
  --agent.wandb-project parahand