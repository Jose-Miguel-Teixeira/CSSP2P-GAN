import torch
import torch.nn as nn
from torchvision.models.resnet import (
    ResNet101_Weights,
    ResNet50_Weights,
    ResNet34_Weights,
    ResNet18_Weights,
)
from typing import (
    Literal,
    List,
    Optional,
    Tuple,
    )

# Project Imports
# from resnet import (
#     ResNet,
#     BasicBlock,
#     Bottleneck,
# )
# from attention import CrossAttentionBlock
from models.resnet import (
    ResNet,
    BasicBlock,
    Bottleneck,
)
from models.attention import CrossAttentionBlock

###############################################################################
# UNet Modules
###############################################################################


class DoubleConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            norm_layer: torch.nn = nn.BatchNorm2d,
    ) -> None:
        super(DoubleConvBlock, self).__init__()

        if not mid_channels:
            mid_channels = out_channels

        if not isinstance(norm_layer, type) or not issubclass(norm_layer, nn.Module):
            raise ValueError("'norm_layer' must be an instance of torch.nn.")

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1
                ),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
                ),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpconvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm_layer: torch.nn = nn.BatchNorm2d,
            use_bilinear: bool = True,
            apply_cross_attention: bool = False,
            embed_num_positions: int = None,
            **kwargs
    ) -> None:
        super(UpconvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_layer = norm_layer
        self.use_bilinear = use_bilinear
        self.apply_cross_attention = apply_cross_attention

        if apply_cross_attention and embed_num_positions is None:
            raise ValueError(
                "'embed_num_positions' must be provided when 'apply_cross_attention' is True."
                )

        if not isinstance(norm_layer, type) or not issubclass(norm_layer, nn.Module):
            raise ValueError("'norm_layer' must be an instance of torch.nn.")

        if use_bilinear:
            self.up = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True
                    ),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=1
                    ),
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=2,
                stride=2,
                )

        if apply_cross_attention:
            self.cross_attn = CrossAttentionBlock(
                channels=in_channels // 2,
                embed_dim=in_channels,
                embed_num_positions=embed_num_positions,
                num_heads=in_channels // 128,
                dropout=kwargs.get('dropout', 0.1),
                use_learnable_positional_embeddings=kwargs.get('use_learnable_positional_embeddings', False),
                ffn_hidden_dim=in_channels * 2,  # ¿ 'in_channels' or 'in_channels * 2' ?
                attention_dropout=kwargs.get('attention_dropout', 0.0),
            )

        self.double_conv = DoubleConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_layer=norm_layer,
        )

    def forward(
            self,
            x1: torch.Tensor,  # Upsampled
            x2: torch.Tensor,  # Skip Connection
    ) -> torch.Tensor:
        x1 = self.up(x1)

        if self.apply_cross_attention:
            # * Construct Mask
            # The key_padding_mask is designed to mark positions in the key
            # that should be ignored during the attention computation, which
            # in this case corresponds to the padding mask.

            # key_padding_mask = torch.zeros_like(x1, dtype=torch.bool)
            # key_padding_mask[:, :, :diffY // 2, :] = True
            # key_padding_mask[:, :, -diffY + diffY // 2:, :] = True
            # key_padding_mask[:, :, :, :diffX // 2] = True
            # key_padding_mask[:, :, :, -diffX + diffX // 2:] = True

            # ** Reshape Mask
            # B, _, W, H = key_padding_mask.shape
            # key_padding_mask = key_padding_mask[:, 0, :, :].reshape(B, H * W)  # [B, H * W]

            # ! In the present case the key_padding_mask is not needed as we
            # ! don't have padding.

            key_padding_mask = None
            x2 = self.cross_attn(
                query=x1,
                key=x2,
                key_padding_mask=key_padding_mask,
                )

        x = torch.cat([x2, x1], dim=1)

        return self.double_conv(x)


class OutConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ) -> None:
        super(OutConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            )

    def forward(self, x):
        return self.conv(x)

###############################################################################
# ResUNet
###############################################################################


class ResNetEncoder(nn.Module):

    def __init__(
            self,
            pretrained: bool = True,
            norm_layer: Literal['batch', 'instance'] = 'batch',
            activation: Literal['relu', 'leaky_relu', 'elu'] = 'relu',
            backbone: Literal[
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101"
                ] = "resnet50",
            ) -> None:
        super(ResNetEncoder, self).__init__()

        # * Define Normalization layer
        match norm_layer:
            case "batch":
                norm_layer = nn.BatchNorm2d
            case "instance":
                norm_layer = nn.InstanceNorm2d
            case _:
                raise ValueError(
                    "Invalid Norm Layer! Choose from 'batch', 'instance'."
                    )

        # * Define Activation function
        match activation:
            case "relu":
                activation = nn.ReLU
            case "leaky_relu":
                activation = nn.LeakyReLU
            case "elu":
                activation = nn.ELU
            case _:
                raise ValueError(
                    "Invalid Activation! Choose from 'relu', 'leaky_relu'."
                    )

        # * Define Backbone
        match backbone:
            case "resnet18":
                self.model = ResNet(
                    block=BasicBlock,
                    layers=[2, 2, 2, 2],
                    norm_layer=norm_layer,
                    )

                self.model.conv1.stride = (1, 1)
                self._substitute_activation_function(
                    model=self.model,
                    activation=activation
                    )

                if pretrained:
                    weights = ResNet18_Weights.DEFAULT
                    state_dict = weights.get_state_dict(progress=True, check_hash=True)

                    # Remove running stats for InstanceNorm2d
                    if norm_layer != nn.BatchNorm2d:
                        keys_to_remove = [key for key in state_dict.keys() if 'running_mean' in key or 'running_var' in key]
                        for key in keys_to_remove:
                            del state_dict[key]

                    self.model.load_state_dict(
                        state_dict=state_dict,
                        strict=True if norm_layer == nn.BatchNorm2d else False,
                    )

            case "resnet34":
                self.model = ResNet(
                    block=BasicBlock,
                    layers=[3, 4, 6, 3],
                    norm_layer=norm_layer,
                    )

                self.model.conv1.stride = (1, 1)
                self._substitute_activation_function(
                    model=self.model,
                    activation=activation
                    )

                if pretrained:
                    weights = ResNet34_Weights.DEFAULT
                    state_dict = weights.get_state_dict(progress=True, check_hash=True)

                    # Remove running stats for InstanceNorm2d
                    if norm_layer != nn.BatchNorm2d:
                        keys_to_remove = [key for key in state_dict.keys() if 'running_mean' in key or 'running_var' in key]
                        for key in keys_to_remove:
                            del state_dict[key]

                    self.model.load_state_dict(
                        state_dict=state_dict,
                        strict=True if norm_layer == nn.BatchNorm2d else False,
                    )

            case "resnet50":
                self.model = ResNet(
                    block=Bottleneck,
                    layers=[3, 4, 6, 3],
                    norm_layer=norm_layer,
                    )

                self.model.conv1.stride = (1, 1)
                self._substitute_activation_function(
                    model=self.model,
                    activation=activation
                    )

                if pretrained:
                    weights = ResNet50_Weights.DEFAULT
                    state_dict = weights.get_state_dict(progress=True, check_hash=True)

                    # Remove running stats for InstanceNorm2d
                    if norm_layer != nn.BatchNorm2d:
                        keys_to_remove = [key for key in state_dict.keys() if 'running_mean' in key or 'running_var' in key]
                        for key in keys_to_remove:
                            del state_dict[key]

                    self.model.load_state_dict(
                        state_dict=state_dict,
                        strict=True if norm_layer == nn.BatchNorm2d else False,
                    )

            case "resnet101":
                self.model = ResNet(
                    block=Bottleneck,
                    layers=[3, 4, 23, 3],
                    norm_layer=norm_layer,
                    )

                self.model.conv1.stride = (1, 1)
                self._substitute_activation_function(
                    model=self.model,
                    activation=activation
                    )

                if pretrained:
                    weights = ResNet101_Weights.DEFAULT
                    state_dict = weights.get_state_dict(progress=True, check_hash=True)

                    # Remove running stats for InstanceNorm2d
                    if norm_layer != nn.BatchNorm2d:
                        keys_to_remove = [key for key in state_dict.keys() if 'running_mean' in key or 'running_var' in key]
                        for key in keys_to_remove:
                            del state_dict[key]

                    self.model.load_state_dict(
                        state_dict=state_dict,
                        strict=True if norm_layer == nn.BatchNorm2d else False,
                    )
            case _:
                raise ValueError(
                    "Invalid Encoder!"
                    "Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101'."
                    )

    def _substitute_activation_function(
            self,
            model: nn.Module,
            activation: nn.Module
            ) -> None:
        for name, module in model.named_children():
            if isinstance(module, nn.ReLU):  # Original activation is ReLU
                setattr(model, name, activation(inplace=True))
            else:
                self._substitute_activation_function(module, activation)

    def forward(self, x) -> Tuple[torch.Tensor, List]:
        SkipConnections = [None] * 4

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        SkipConnections[0] = x

        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        SkipConnections[1] = x

        x = self.model.layer2(x)
        SkipConnections[2] = x

        x = self.model.layer3(x)
        SkipConnections[3] = x

        x = self.model.layer4(x)

        return x, SkipConnections


class ResNetDecoder(nn.Module):
    def __init__(
            self,
            input_size: Tuple,
            output_channels: int,
            hidden_sizes: List[int],
            backbone: Literal["resnet34", "resnet50", "resnet101"],
            norm_layer: Literal['batch', 'instance'] = 'batch',
            use_bilinear: bool = True,
            attention_layers: List[int] = None,
            **kwargs
    ) -> None:
        super(ResNetDecoder, self).__init__()

        self.input_size = input_size
        self.output_channels = output_channels
        self.attention_layers = attention_layers
        self.backbone = backbone
        self.use_bilinear = use_bilinear

        # * Define Normalization Layer
        match norm_layer:
            case "batch":
                self.norm_layer = nn.BatchNorm2d
            case "instance":
                self.norm_layer = nn.InstanceNorm2d
            case _:
                raise ValueError(
                    "Invalid Norm Layer! Choose from 'batch', 'instance'."
                    )

        # * Define Decoder Layers
        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_sizes)):
            if i == 4:
                if backbone in ("resnet34", "resnet18"):
                    self.layers.append(
                        nn.Conv2d(
                            in_channels=hidden_sizes[i],
                            out_channels=hidden_sizes[i] // 2,
                            kernel_size=1,
                            stride=1,
                        )
                    )
                else:
                    self.layers.append(
                        nn.Conv2d(
                            in_channels=hidden_sizes[i],
                            out_channels=hidden_sizes[i] * 2,
                            kernel_size=1,
                            stride=1,
                        )
                    )
            else:
                self.layers.append(
                    nn.Conv2d(
                        in_channels=hidden_sizes[i],
                        out_channels=hidden_sizes[i],
                        kernel_size=1,
                        stride=1,
                    )
                )

            if attention_layers is not None and (i - 1) in attention_layers:
                embed_num_positions = int(
                    (input_size[0] // 2 ** (len(hidden_sizes) - (i + 1))) ** 2
                )
                self.layers.append(
                    UpconvBlock(
                        in_channels=hidden_sizes[i - 1],
                        out_channels=hidden_sizes[i],
                        norm_layer=self.norm_layer,
                        use_bilinear=self.use_bilinear,
                        apply_cross_attention=True,
                        embed_num_positions=embed_num_positions,
                        **kwargs
                    )
                )
            else:
                self.layers.append(
                    UpconvBlock(
                        in_channels=hidden_sizes[i - 1],
                        out_channels=hidden_sizes[i],
                        norm_layer=self.norm_layer,
                        use_bilinear=self.use_bilinear,
                        apply_cross_attention=False,
                    )
                )
        self.layers.append(
            OutConvBlock(
                hidden_sizes[-1],
                out_channels=output_channels,
                )
            )

    def forward(
            self,
            x: torch.Tensor,
            skip_connections: List[torch.Tensor],
    ) -> torch.Tensor:

        reversed_skip_connections = skip_connections[::-1]

        for i in range(0, len(self.layers) - 1, 2):
            # Debug
            # print('Layer: ', i)
            # print('Input: ', x.shape)

            skip_connection = reversed_skip_connections[i // 2]

            # Debug
            # print('Skip Connection: ', skip_connection.shape)
            # print(self.layers[i])
            # print(self.layers[i + 1])
            skip_connection = self.layers[i](skip_connection)

            # Debug
            # print('Transformed Skip Connection: ', skip_connection.shape)

            x = self.layers[i + 1](x, skip_connection)
        else:
            x = self.layers[-1](x)
        return x


class ResUNet(nn.Module):
    def __init__(
            self,
            input_size: Tuple,
            output_channels: int,
            final_activation: Literal['sigmoid', 'tanh', 'none'] = 'sigmoid',
            encoder_activation: Literal['relu', 'leaky_relu', 'elu'] = 'relu',
            norm_layer: Literal['batch', 'instance'] = 'batch',
            use_bilinear: bool = True,
            backbone: Literal[
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101"
                ] = "resnet50",
            attention_mechanism: Optional[
                List[
                    Literal[
                        'cross-attention'
                        ]
                    ]
                ] = None,
            **kwargs
            ) -> None:
        super(ResUNet, self).__init__()

        if input_size[0] != input_size[1]:
            raise ValueError(
                "Input size must have the same height and width."
                f"Got: {input_size}."
                )

        # * Define Decoder
        match backbone:
            case "resnet18":
                hidden_sizes = [512, 256, 128, 64, 64]
            case "resnet34":
                hidden_sizes = [512, 256, 128, 64, 64]
            case "resnet50":
                hidden_sizes = [2048, 1024, 512, 256, 64]
            case "resnet101":
                hidden_sizes = [2048, 1024, 512, 256, 64]
            case _:
                raise ValueError(
                    "Invalid Encoder!"
                    "Choose from 'resnet18' 'resnet34', 'resnet50', 'resnet101'."
                    )

        if attention_mechanism is not None:
            if isinstance(attention_mechanism, str):
                attention_mechanism = [attention_mechanism]
            elif not isinstance(attention_mechanism, list):
                raise ValueError(
                    "'attention_mechanism' must be a list of strings."
                )
            valid_attention_mechanisms = ['cross-attention']
            for mechanism in attention_mechanism:
                if mechanism not in valid_attention_mechanisms:
                    raise ValueError(
                        f"Invalid Attention Mechanism: {mechanism}! "
                        f"Choose from {valid_attention_mechanisms}."
                        )
                if mechanism == 'cross-attention':
                    attention_layers = kwargs.pop('attention_layers', None)
                    if attention_layers is None:
                        valid_attention_layers = [i for i in range(len(hidden_sizes))]
                        raise ValueError(
                            "'attention_layers' must be provided for 'cross-attention'."
                            f"Choose {valid_attention_layers} or a subset of it."
                            )
        else:
            kwargs.pop('attention_layers', None)
            attention_layers = None

        self.encoder = ResNetEncoder(
            pretrained=True,
            norm_layer=norm_layer,
            backbone=backbone,
            activation=encoder_activation,
            )

        self.decoder = ResNetDecoder(
            input_size=input_size,
            output_channels=output_channels,
            hidden_sizes=hidden_sizes,
            norm_layer=norm_layer,
            use_bilinear=use_bilinear,
            backbone=backbone,
            attention_layers=attention_layers,
            **kwargs
            )

        # * Define Final Activation Function
        match final_activation:
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case "tanh":
                self.activation = nn.Tanh()
            case "none":
                self.activation = nn.Identity()
            case _:
                raise ValueError(
                    "Invalid Activation! Choose from 'sigmoid', 'tanh'."
                    )

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        x = self.activation(x)
        return x


def dummy_training(
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device('cpu'),
        run_anomaly_detection: bool = False,
        num_examples: int = 1000,
        ) -> None:

    HE = torch.rand(num_examples, 1, 3, 512, 512)
    IHC = torch.rand(num_examples, 1, 3, 512, 512)
    total_time: float = 0.0

    if device.type == 'cuda':
        scaler = GradScaler()

    model.train()
    model = model.to(device)
    if run_anomaly_detection:
        print("Running anomaly detection ...\n")
        with torch.autograd.detect_anomaly():
            for i, (x, y) in enumerate(zip(HE, IHC)):
                tic = time()
                x, y = x.to(device), y.to(device)
                toc = time()
                print(f"Time taken for data transfer: {toc - tic:.4f} seconds.")
                optimizer.zero_grad()
                if i == 0:
                    tic = time()
                    print("Model Compilation ...\n")
                    if device.type == 'cuda':
                        with autocast():
                            y_hat = model(x)
                            loss = criterion(y_hat, y)
                            scaler.scale(loss).backward()
                    else:
                        y_hat = model(x)
                        loss = criterion(y_hat, y)
                        loss.backward()
                    optimizer.step()
                    toc = time()
                    print(f"Time taken for model compilation: {toc - tic:.4f} seconds.")
                else:
                    tic = time()
                    if device.type == 'cuda':
                        with autocast():
                            y_hat = model(x)
                            loss = criterion(y_hat, y)
                            scaler.scale(loss).backward()
                    else:
                        y_hat = model(x)
                        loss = criterion(y_hat, y)
                        loss.backward()
                    optimizer.step()
                    toc = time()
                    print(f"Time taken for forward pass: {toc - tic:.4f} seconds.")
                    total_time += toc - tic
        print(f"Total time taken: {total_time:.4f} seconds.")
        print("Anomaly detection successful.\n")
    else:
        for i, (x, y) in enumerate(zip(HE, IHC)):
            tic = time()
            x, y = x.to(device), y.to(device)
            toc = time()
            print(f"Time taken for data transfer: {toc - tic:.4f} seconds.")
            optimizer.zero_grad()
            if i == 0:
                tic = time()
                print("Model Compilation ...")
                if device.type == 'cuda':
                    with autocast():
                        y_hat = model(x)
                        loss = criterion(y_hat, y)
                        scaler.scale(loss).backward()
                else:
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    optimizer.zero_grad()
                    loss.backward()
                optimizer.step()
                toc = time()
                print(f"Time taken for model compilation: {toc - tic:.4f} seconds.")
            else:
                tic = time()
                if device.type == 'cuda':
                    with autocast():
                        y_hat = model(x)
                        loss = criterion(y_hat, y)
                        scaler.scale(loss).backward()
                else:
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    optimizer.zero_grad()
                    loss.backward()
                optimizer.step()
                toc = time()
                print(f"Time taken for forward pass: {toc - tic:.4f} seconds.")
                total_time += toc - tic
            loss = criterion(y_hat, y)
    print(f"Total time for dummy training: {total_time:.4f} seconds.")
    return None


if __name__ == "__main__":
    from torchinfo import summary
    from time import time
    from torch.cuda.amp import autocast, GradScaler

    # import torch._dynamo
    # torch._dynamo.config.suppress_errors = True  # fallback to eager mode
    # torch._dynamo.config.capture_scalar_outputs = True  # capture scalar outputs

    print("Flash SDP enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("Memory efficient SDP enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("Math SDP enabled:", torch.backends.cuda.math_sdp_enabled())

    # * Define Model
    model = ResUNet(
        input_size=(512, 512),
        output_channels=3,
        encoder_activation='relu',
        final_activation='sigmoid',
        norm_layer='instance',
        backbone='resnet18',
        use_bilinear=True,
        attention_mechanism='cross-attention',
        attention_layers=[0, 1],
        use_learnable_positional_embeddings=False,
        dropout=0.2,
        attention_dropout=0.1,
    )

    # * Print Model Summary
    print("Model Summary:")
    summary(model, input_size=(1, 3, 512, 512))

    # * Compile Model
    model = torch.compile(model, backend="inductor", fullgraph=True)
    # To visualize graph breaks export: TORCH_LOGS="graph_breaks"

    # * Define Criterion
    criterion = nn.MSELoss()

    # * Define Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tic = time()
    dummy_training(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device('cuda'),
        run_anomaly_detection=True,
        num_examples=1000,
        )
    toc = time()
    print(f"Total time taken: {toc - tic:.4f} seconds.")
