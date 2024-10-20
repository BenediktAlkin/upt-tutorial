import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from torch import nn
from torch_geometric.nn.pool import radius_graph
from torch_scatter import segment_csr


class SupernodePooling(nn.Module):
    def __init__(
            self,
            radius,
            max_degree,
            input_dim,
            hidden_dim,
            ndim,
            init_weights="torch",
    ):
        super().__init__()
        self.radius = radius
        self.max_degree = max_degree
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.init_weights = init_weights

        self.input_proj = LinearProjection(input_dim, hidden_dim, init_weights=init_weights)
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        self.message = nn.Sequential(
            LinearProjection(hidden_dim * 2, hidden_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(hidden_dim, hidden_dim, init_weights=init_weights),
        )
        self.output_dim = hidden_dim

    def forward(self, input_feat, input_pos, supernode_idxs, batch_idx):
        assert input_feat.ndim == 2
        assert input_pos.ndim == 2
        assert supernode_idxs.ndim == 1

        # radius graph
        input_edges = radius_graph(
            x=input_pos,
            r=self.radius,
            max_num_neighbors=self.max_degree,
            batch=batch_idx,
            loop=True,
            # inverted flow direction is required to have sorted dst_indices
            flow="target_to_source",
        )
        is_supernode_edge = torch.isin(input_edges[0], supernode_idxs)
        input_edges = input_edges[:, is_supernode_edge]

        # embed mesh
        x = self.input_proj(input_feat) + self.pos_embed(input_pos)

        # create message input
        dst_idx, src_idx = input_edges.unbind()
        x = torch.concat([x[src_idx], x[dst_idx]], dim=1)
        x = self.message(x)
        # accumulate messages
        # indptr is a tensor of indices betweeen which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = dst_idx.unique(return_counts=True)
        # first index has to be 0
        # NOTE: padding for target indices that dont occour is not needed as self-loop is always present
        padded_counts = torch.zeros(len(counts) + 1, device=counts.device, dtype=counts.dtype)
        padded_counts[1:] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")

        # sanity check: dst_indices has len of batch_size * num_supernodes and has to be divisible by batch_size
        # if num_supernodes is not set in dataset this assertion fails
        batch_size = batch_idx.max() + 1
        assert dst_indices.numel() % batch_size == 0

        # convert to dense tensor (dim last)
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )

        return x
