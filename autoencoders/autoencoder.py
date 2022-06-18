import torch
import torch.nn as nn


def get_dense_mnist_autoencoder(encoder_dropout: float = 0.0, decoder_dropout: float = 0.0) -> nn.Module:
    """
    :param encoder_dropout: Amount of dropout to add after each encoder layer in range [0,1].
    :param decoder_dropout: Amount of dropout to add after each decoder layer in range [0,1].
    """

    class Model(nn.Module):
        def __init__(self, encoder_dropout: float, decoder_dropout: float):
            super(Model, self).__init__()

            input_size = 784  # Input is flattened image of size 28 * 28

            self._encoder = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Dropout(encoder_dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(encoder_dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(encoder_dropout),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(encoder_dropout),
            )

            self._decoder = nn.Sequential(
                nn.Linear(32, 128),
                nn.ReLU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(decoder_dropout),
                nn.Linear(512, input_size),
                nn.Sigmoid(),  # Want output values in range [0,1] (same as input)
            )

        def forward(self, x):
            x = torch.flatten(x, 1)  # Flatten image
            x = self._encoder(x)
            x = self._decoder(x)
            x = torch.reshape(x, (len(x), 1, 28, 28))  # Reconstruct image
            return x

    return Model(encoder_dropout, decoder_dropout)


def get_cnn_mnist_autoencoder() -> nn.Module:
    pass
