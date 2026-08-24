import os
import torch

# 1. Robust Weight Loading for Evaluation
checkpoint_path = "weight/checkpoint_epoch_10.pt"  # Update to your best checkpoint

if os.path.exists(checkpoint_path):
    print(f"Loading evaluation checkpoint from '{checkpoint_path}'...")
    # map_location=device is critical here if you trained on GPU but are evaluating on CPU/another GPU
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    epoch = checkpoint.get("epoch", "Unknown")
    loss = checkpoint.get("loss", "Unknown")
    print(f"Successfully loaded weights from epoch {epoch} (Training Loss: {loss}).")
else:
    print(f"WARNING: Checkpoint '{checkpoint_path}' not found!")
    print("Evaluating with UNTRAINED (random) weights. The generated actions will be noise.")

model.eval()
with torch.no_grad():
    batch = next(iter(loader))
    ego_hist = batch["ego_history"][:1].to(device)
    agents = batch["agents_history"][:1].to(device)
    map_lines = batch["map_lines"][:1].to(device)
    agent_mask = batch["agent_mask"][:1].to(device)
    map_mask = batch["map_mask"][:1].to(device)

    proprio = torch.cat([
        ego_hist[:, -1, :],
        torch.zeros((1, config.dim_y - 6), device=device)
    ], dim=-1)

    enc_out = model.fallback_encoder(ego_hist, agents, map_lines, agent_mask, map_mask)

    gen_actions = model.generate(
        diffusion_sde=diffusion_sde,
        encoder_hidden_states=enc_out.last_hidden_state,
        proprio=proprio,
        attention_mask=enc_out.attention_mask,
        steps=6,
    )
    print(f"Generated actions shape (DPM-Solver): {gen_actions.shape}")
