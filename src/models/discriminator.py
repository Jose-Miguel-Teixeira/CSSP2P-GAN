import torch
import torch.nn as nn
from typing import Literal
from models.networks import Downsample  # models.networks to run in train_gan.py


# NLayerDiscriminator from Fangda Li's repository
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
            self,
            input_nc: int,
            ndf: int = 64,
            n_layers: int = 3,
            norm_layer: Literal['instance', 'batch', 'none', None] = 'batch',
            no_antialias: bool = False,
            weight_norm: Literal['spectral', 'none', None] = 'spectral',
            ) -> None:
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        if norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        elif norm_layer == 'none' or norm_layer is None:
            norm_layer = nn.Identity
            use_bias = True
        else:
            raise NotImplementedError(
                'Discriminator normalization layer [%s] is not found.' % norm_layer
                )

        if weight_norm == 'spectral':
            weight_norm = nn.utils.spectral_norm
        elif weight_norm == 'none' or weight_norm is None:
            def weight_norm(x): return x
        else:
            raise NotImplementedError(
                'Discriminator weight normalization [%s] is not found.' % weight_norm
                )

        kw = 4
        padw = 1

        if no_antialias:
            sequence = [
                nn.Conv2d(
                    in_channels=input_nc,
                    out_channels=ndf,
                    kernel_size=kw,
                    stride=2,
                    padding=padw
                    ),
                nn.LeakyReLU(
                    inplace=True,
                    negative_slope=0.2
                    )
                ]
        else:
            sequence = [
                nn.Conv2d(
                    in_channels=input_nc,
                    out_channels=ndf,
                    kernel_size=kw,
                    stride=1,
                    padding=padw
                    ),
                nn.LeakyReLU(
                    inplace=True,
                    negative_slope=0.2
                    ),
                Downsample(ndf)
                ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(
                        in_channels=ndf * nf_mult_prev,
                        out_channels=ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias
                        ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(
                        inplace=True,
                        negative_slope=0.2
                        )
                ]
            else:
                sequence += [
                    nn.Conv2d(
                        in_channels=ndf * nf_mult_prev,
                        out_channels=ndf * nf_mult,
                        kernel_size=kw,
                        stride=1,
                        padding=padw,
                        bias=use_bias
                        ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(
                        inplace=True,
                        negative_slope=0.2
                        ),
                    Downsample(ndf * nf_mult)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                in_channels=ndf * nf_mult_prev,
                out_channels=ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias
                ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(
                inplace=True,
                negative_slope=0.2
                )
        ]

        for i, layer in enumerate(sequence):
            if isinstance(layer, nn.Conv2d):
                sequence[i] = weight_norm(layer)

        self.enc = nn.Sequential(*sequence)

        # output 1 channel prediction map
        self.final_conv = weight_norm(
            nn.Conv2d(
                in_channels=ndf * nf_mult,
                out_channels=1,
                kernel_size=kw,
                stride=1,
                padding=padw
                )
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Standard forward."""
        final_ft = self.enc(input)
        dout = self.final_conv(final_ft)
        return dout


def dummy_training(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cpu'),
        run_anomaly_detection: bool = False,
        num_examples: int = 100
        ) -> None:

    x = torch.rand(num_examples, 5, 3, 512, 512)
    y = torch.randint(0, 1, (num_examples, 5, 1, 62, 62)).to(torch.float32)

    model.to(device)
    model.train()
    total_time: float = 0.0

    if run_anomaly_detection:
        print('Running anomaly detection...')
        with torch.autograd.detect_anomaly():
            for i, (x, y) in enumerate(zip(x, y)):
                x, y = x.to(device), y.to(device)
                if i == 0:
                    print("Model Compilation ...\n")
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print("Model Compilation Done!\n")
                    print("\nForward pass successful.\n")
                else:
                    tic = time()
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    toc = time()
                    print(f"Time taken for forward pass: {toc - tic:.4f} seconds.")
                    total_time += toc - tic
            else:
                print(f"Total time taken: {total_time:.4f} seconds.")
                print('Anomaly detection completed!')
    else:
        print('Running training loop...')
        for i, (x, y) in enumerate(zip(x, y)):
            x, y = x.to(device), y.to(device)
            if i == 0:
                print("Model Compilation ...\n")
                y_hat = model(x)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("Model Compilation Done!\n")
                print("\nForward pass successful.\n")
            else:
                tic = time()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                toc = time()
                print(f"Time taken for forward pass: {toc - tic:.4f} seconds.")
                total_time += toc - tic
        else:
            print(f"Total time taken: {total_time:.4f} seconds.")
            print('Training loop completed!')


if __name__ == '__main__':
    from torchinfo import summary
    from time import time

    model = NLayerDiscriminator(
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_layer='batch',
        no_antialias=False,
        weight_norm='spectral'
        )

    # * Model Summary
    summary(model, input_size=(1, 3, 512, 512), device='cpu')

    # * Model Compilation
    model = torch.compile(model, backend='inductor', fullgraph=True)

    # * Define Criterion
    criterion = nn.BCEWithLogitsLoss()

    # * Define Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    dummy_training(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device('cuda'),
        run_anomaly_detection=True
    )
