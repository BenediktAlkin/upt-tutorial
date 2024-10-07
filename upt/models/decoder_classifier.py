from functools import partial

from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import DitBlock
from kappamodules.vit import VitBlock
from torch import nn


class DecoderClassifier(nn.Module):
    def __init__(
            self,
            input_dim,
            num_classes,
            dim,
            depth,
            num_heads,
            cond_dim=None,
            init_weights="truncnormal002",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights

        # input projection
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)

        # blocks
        if cond_dim is None:
            block_ctor = VitBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(
                    dim=dim,
                    num_heads=num_heads,
                    init_weights=init_weights,
                )
                for _ in range(depth)
            ],
        )

        # classifier
        self.head = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            LinearProjection(dim, num_classes),
        )

    def forward(self, x, condition=None):
        # check inputs
        assert x.ndim == 3, "expected shape (batch_size, num_latent_tokens, dim)"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # input projection
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        # pool
        x = x.mean(dim=1)

        # classify
        x = self.head(x)

        return x
