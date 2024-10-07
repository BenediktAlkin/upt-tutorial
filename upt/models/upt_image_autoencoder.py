from torch import nn


class UPTImageAutoencoder(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()
        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder

    def forward(self, x, output_pos):
        # encode data
        latent = self.encoder(x)

        # propagate forward
        latent = self.approximator(latent)

        # decode
        pred = self.decoder(latent, output_pos=output_pos)

        return pred
