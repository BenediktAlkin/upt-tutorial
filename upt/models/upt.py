import einops
import torch
from torch import nn


class UPT(nn.Module):
    def __init__(self, conditioner, encoder, approximator, decoder):
        super().__init__()
        self.conditioner = conditioner
        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder

    def forward(
            self,
            input_feat,
            input_pos,
            supernode_idxs,
            output_pos,
            batch_idx,
            timestep,
    ):
        condition = self.conditioner(timestep)

        # encode data
        latent = self.encoder(
            input_feat=input_feat,
            input_pos=input_pos,
            supernode_idxs=supernode_idxs,
            batch_idx=batch_idx,
            condition=condition,
        )

        # propagate forward
        latent = self.approximator(latent, condition=condition)

        # decode
        pred = self.decoder(
            x=latent,
            output_pos=output_pos,
            condition=condition,
        )

        return pred

    @torch.no_grad()
    def rollout(self, input_feat, input_pos, supernode_idxs, batch_idx):
        batch_size = batch_idx.max() + 1
        timestep = torch.zeros(batch_size).long()

        # we assume that output_pos is simply a rearranged input_pos
        # i.e. num_inputs == num_outputs and num_inputs is constant for all samples
        output_pos = einops.rearrange(
            input_pos,
            "(batch_size num_inputs) ndim -> batch_size num_inputs ndim",
            batch_size=batch_size,
        )

        predictions = []
        for i in range(self.conditioner.num_timesteps):
            condition = self.conditioner(timestep)
            # encode data
            latent = self.encoder(
                input_feat=input_feat,
                input_pos=input_pos,
                supernode_idxs=supernode_idxs,
                batch_idx=batch_idx,
                condition=condition,
            )

            # propagate forward
            latent = self.approximator(latent, condition=condition)

            # decode
            pred = self.decoder(
                x=latent,
                output_pos=output_pos,
                condition=condition,
            )
            predictions.append(pred)

            # increase timestep
            timestep += 1

            # feed prediction as next input for autoregressive rollout
            input_feat = pred

        return predictions
