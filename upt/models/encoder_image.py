from functools import partial

import einops
from kappamodules.layers import Sequential
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, DitPerceiverPoolingBlock, DitBlock
from kappamodules.utils.param_checking import to_2tuple
from kappamodules.vit import VitPatchEmbed, VitPosEmbed2d
from torch import nn


class EncoderImage(nn.Module):
    def __init__(
            self,
            input_dim,
            patch_size,
            resolution,
            enc_dim,
            enc_num_heads,
            enc_depth,
            perc_dim=None,
            perc_num_heads=None,
            num_latent_tokens=None,
            cond_dim=None,
            init_weights="truncnormal",
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        resolution = to_2tuple(resolution)
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.resolution = resolution
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.num_latent_tokens = num_latent_tokens
        self.condition_dim = cond_dim
        self.init_weights = init_weights

        # embed
        self.patch_embed = VitPatchEmbed(
            dim=enc_dim,
            num_channels=input_dim,
            resolution=resolution,
            patch_size=patch_size,
        )
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=enc_dim, is_learnable=False)

        # blocks
        if cond_dim is None:
            block_ctor = PrenormBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(dim=enc_dim, num_heads=enc_num_heads, init_weights=init_weights)
                for _ in range(enc_depth)
            ],
        )

        # perceiver pooling
        if num_latent_tokens is None:
            self.perceiver = None
        else:
            if cond_dim is None:
                block_ctor = partial(
                    PerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        init_weights=init_weights,
                    ),
                )
            else:
                block_ctor = partial(
                    DitPerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        cond_dim=cond_dim,
                        init_weights=init_weights,
                    ),
                )
            self.perceiver = block_ctor(
                dim=perc_dim,
                num_heads=perc_num_heads,
                num_query_tokens=num_latent_tokens,
            )

    def forward(self, input_image, condition=None):
        # check inputs
        assert input_image.ndim == 4, "expected input image of shape (batch_size, num_channels, height, width)"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # patch_embed
        x = self.patch_embed(input_image)
        # add pos_embed
        x = self.pos_embed(x)
        # flatten
        x = einops.rearrange(x, "b ... d -> b (...) d")

        # transformer
        x = self.blocks(x, **cond_kwargs)

        # perceiver
        if self.perceiver is not None:
            x = self.perceiver(kv=x, **cond_kwargs)

        return x
