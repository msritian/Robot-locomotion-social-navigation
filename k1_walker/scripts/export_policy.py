"""Export an RSL-RL actor checkpoint to TorchScript without launching Isaac Sim (PROJECT_SPEC B.5 item 5).
policy(obs[N, 235]) -> leg actions[N, 12]; includes the empirical observation normalizer
(rsl_rl.networks.normalization: (x - mean) / (std + 1e-2)). Hidden sizes/activation from params/agent.yaml.

    python export_policy.py model_3998.pt k1_walker.pt
"""
import sys

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, sd):
        super().__init__()
        ws = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
        layers = []
        for i, idx in enumerate(ws):
            w, b = sd[f"actor.{idx}.weight"], sd[f"actor.{idx}.bias"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers.append(lin)
            if i < len(ws) - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)
        self.register_buffer("mean", sd["actor_obs_normalizer._mean"].clone())
        self.register_buffer("std", sd["actor_obs_normalizer._std"].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net((x - self.mean) / (self.std + 1e-2)), -5.0, 5.0)   # = clip_actions in training


if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    actor = Actor(ck["model_state_dict"]).eval()
    ts = torch.jit.script(actor)
    ts.save(sys.argv[2])
    x = torch.zeros(1, actor.mean.shape[1])
    print("exported", sys.argv[2], "iter", ck.get("iter"), "in", tuple(x.shape), "out", tuple(ts(x).shape),
          "torch", torch.__version__)
