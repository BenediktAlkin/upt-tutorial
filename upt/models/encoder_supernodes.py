from functools import partial

from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, DitPerceiverPoolingBlock, DitBlock
from upt.modules.supernode_pooling import SupernodePooling
from torch import nn


class EncoderSupernodes(nn.Module):
    def __init__(
            self,
            input_dim,
            ndim,
            radius,
            max_degree,
            gnn_dim,
            enc_dim,
            enc_depth,
            enc_num_heads,
            perc_dim=None,
            perc_num_heads=None,
            num_latent_tokens=None,
            cond_dim=None,
            init_weights="truncnormal",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ndim = ndim
        self.radius = radius
        self.max_degree = max_degree
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.num_latent_tokens = num_latent_tokens
        self.condition_dim = cond_dim
        self.init_weights = init_weights

        # supernode pooling
        self.supernode_pooling = SupernodePooling(
            radius=radius,
            max_degree=max_degree,
            input_dim=input_dim,
            hidden_dim=gnn_dim,
            ndim=ndim,
            init_weights=init_weights,
        )

        # blocks
        self.enc_proj = LinearProjection(gnn_dim, enc_dim, init_weights=init_weights, optional=True)
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

    def forward(self, input_feat, input_pos, supernode_idxs, batch_idx, condition=None):
        # check inputs
        assert input_feat.ndim == 2, "expected sparse tensor (batch_size * num_inputs, input_dim)"
        assert input_pos.ndim == 2, "expected sparse tensor (batch_size * num_inputs, ndim)"
        assert len(input_feat) == len(input_pos), "expected input_feat and input_pos to have same length"
        assert supernode_idxs.ndim == 1, "supernode_idxs is a 1D tensor of indices that are used as supernodes"
        assert batch_idx.ndim == 1, f"batch_idx should be 1D tensor that assigns elements of the input to samples"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # supernode pooling
        x = self.supernode_pooling(
            input_feat=input_feat,
            input_pos=input_pos,
            supernode_idxs=supernode_idxs,
            batch_idx=batch_idx,
        )

        # project to encoder dimension
        x = self.enc_proj(x)

        # transformer
        x = self.blocks(x, **cond_kwargs)

        # perceiver
        if self.perceiver is not None:
            x = self.perceiver(kv=x, **cond_kwargs)

        return x
