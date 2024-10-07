from torch import nn


class UPTImageClassifier(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()
        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder

    def forward(self, x):
        # encode data
        latent = self.encoder(x)

        # propagate forward
        latent = self.approximator(latent)

        # decode
        pred = self.decoder(latent)

        return pred
