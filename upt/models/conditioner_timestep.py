from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
from torch import nn


class ConditionerTimestep(nn.Module):
    def __init__(self, dim, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.dim = dim
        self.register_buffer(
            "timestep_embed",
            get_sincos_1d_from_seqlen(seqlen=num_timesteps, dim=dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, timestep):
        # checks + preprocess
        assert timestep.numel() == len(timestep)
        timestep = timestep.flatten()
        # embed
        embed = self.mlp(self.timestep_embed[timestep])
        return embed
