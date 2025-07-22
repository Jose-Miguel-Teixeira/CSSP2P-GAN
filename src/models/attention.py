# Documentation:
# https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html
# https://pytorch.org/tutorials/prototype/nestedtensor.html
# https://pytorch.org/vision/main/models/vision_transformer.html

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn.attention import sdpa_kernel, SDPBackend
# from time import time


class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            embed_num_positions: int,
            embed_dim: int = 256,
            num_heads: int = 4,
            attention_dropout: float = 0.0,
            dropout: float = 0.1,
            use_learnable_positional_embeddings: bool = True,
            ffn: Optional[nn.Module] = None,
            ffn_hidden_dim: int = 512,
            ffn_activation: nn.Module = nn.GELU(),
            ) -> None:
        super(CrossAttentionBlock, self).__init__()

        self.embed_num_positions = embed_num_positions
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_learnable_positional_embeddings = use_learnable_positional_embeddings

        # * 1. Convolutional layers to embed input features
        # [B, C, H, W] -> [B, embed_dim, H, W]
        self.conv1_query = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0
            )
        self.conv1_key = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0
            )

        # * 2. Layer normalization
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)

        # * 3. Multi-Head Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=attention_dropout,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=embed_dim,
            vdim=embed_dim,
            )

        # * 4. Dropout
        self.dropout = nn.Dropout(dropout)

        # * 5. Layer normalization
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # * 6. Feed-forward network
        if ffn is not None:
            self.ffn = ffn
        else:
            ffn_hidden_dim = ffn_hidden_dim or embed_dim * 2
            if dropout > 0:
                self.ffn = nn.Sequential(
                                nn.Linear(embed_dim, ffn_hidden_dim),
                                ffn_activation,
                                nn.Linear(ffn_hidden_dim, embed_dim),
                                nn.Dropout(dropout)
                            )
            else:
                self.ffn = nn.Sequential(
                                nn.Linear(embed_dim, ffn_hidden_dim),
                                ffn_activation,
                                nn.Linear(ffn_hidden_dim, embed_dim)
                            )

        # * 7. Convolutional layer to project back to input feature dimensions
        self.conv2 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0
            )

        # * Construct the positional embeddings
        if self.use_learnable_positional_embeddings:
            self.pos_embed = nn.Parameter(
                torch.empty(
                    1, self.embed_num_positions, self.embed_dim
                    ).normal_(std=0.02)
                )  # from BERT
        else:
            self.register_buffer(
                'pos_embed',
                self._build_sinusoidal_positional_embedding()
                )

        # * Debugging
        # self.print_config()

    def _build_sinusoidal_positional_embedding(self) -> torch.Tensor:
        # Create sinusoidal positional embeddings as in "Attention Is All You Need"
        pe = torch.zeros(self.embed_num_positions, self.embed_dim)
        position = torch.arange(
            0, self.embed_num_positions,
            dtype=torch.float
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_dim)
            )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: [1, num_positions, embed_dim]

    def print_config(self):
        print(f"Channels: {self.conv1_query.in_channels}")
        print(f"Embedding Dimension: {self.embed_dim}")
        print(f"Number of Heads: {self.num_heads}")
        print(f"Attention Dropout: {self.attn.dropout}")
        print(f"Dropout: {self.dropout.p}")
        print(f"Use Learnable Positional Embeddings: {self.use_learnable_positional_embeddings}")
        print(f"Feed-Forward Network: {self.ffn}")
        print(f"Feed-Forward Hidden Dimension: {self.ffn[0].out_features if isinstance(self.ffn, nn.Sequential) else 'N/A'}")
        print(f"Feed-Forward Activation: {self.ffn[1] if isinstance(self.ffn, nn.Sequential) else 'N/A'}\n\n")

    def _process_input(
            self,
            query: torch.Tensor,
            key: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, _, H, W = query.shape

        # * Check input shapes
        if query.shape != key.shape:
            raise ValueError(
                "Query and key must have the same shape."
                f"Got query shape: {query.shape}, key shape: {key.shape}"
                )

        # * Embed input features
        query = self.conv1_query(query)  # [B, C, H, W] -> [B, embed_dim, H, W]
        key = self.conv1_key(key)  # [B, C, H, W] -> [B, embed_dim, H, W]

        # * Flatten spatial dimensions
        query_flat = query.reshape(B, self.embed_dim, H * W).permute(0, 2, 1)  # [B, H*W, embed_dim]
        key_flat = key.reshape(B, self.embed_dim, H * W).permute(0, 2, 1)  # [B, H*W, embed_dim]

        # * Add positional embeddings (broadcast over batch)
        query_flat = query_flat + self.pos_embed
        key_flat = key_flat + self.pos_embed
        return query_flat, key_flat

    # TODO: Neither of the options can be compiled using backend='inductor'
    # and fullgraph=True. This functions performs a data dependent control
    # which the compiler struggles to handle. As a result, we have a graph
    # break which raises an error.

    # * Option 1
    # @torch._dynamo.disable()
    # @torch.compiler.disable(recursive=False)
    # def _assert_valid_key_padding_mask(
    #         self,
    #         key_padding_mask: torch.Tensor
    #         ) -> Optional[torch.Tensor]:

    #     if key_padding_mask is not None:
    #         if key_padding_mask.dim() == 4:
    #             B, _, H, W = key_padding_mask.shape
    #             if torch.all(key_padding_mask == key_padding_mask[:, 0:1, :, :]).bool():
    #                 key_padding_mask = key_padding_mask[:, 0, :, :].reshape(B, H * W)  # [B, H*W]
    #                 return key_padding_mask
    #             else:
    #                 raise ValueError("Padding mask is not consistent across channels.")
    #         elif key_padding_mask.dim() == 3:
    #             B, H, W = key_padding_mask.shape
    #             key_padding_mask = key_padding_mask.reshape(B, H * W)  # [B, H*W]
    #             return key_padding_mask
    #         elif key_padding_mask.dim() == 2:
    #             return key_padding_mask
    #         else:
    #             raise ValueError("Invalid padding mask dimensions.")
    #     return None

    # * Option 2
    # def _assert_valid_key_padding_mask(self, key_padding_mask: torch.Tensor) -> Optional[torch.Tensor]:
    #     if key_padding_mask is not None:
    #         if key_padding_mask.dim() == 4:
    #             B, _, H, W = key_padding_mask.shape

    #             # Create a predicate as a 0-dim tensor:
    #             pred = torch.equal(key_padding_mask, key_padding_mask[:, 0:1, :, :])

    #             if torch._dynamo.is_compiling():
    #                 # Define the branch functions:
    #                 def true_fn():
    #                     # In the true branch, collapse the channel dimension.
    #                     return key_padding_mask[:, 0, :, :].reshape(B, H * W)

    #                 def false_fn():
    #                     # In the false branch, raise the error.
    #                     raise ValueError("Padding mask is not consistent across channels.")

    #                 # Use torch.cond to choose the branch.
    #                 key_padding_mask = torch.cond(pred, true_fn, false_fn)
    #                 return key_padding_mask
    #             else:
    #                 if pred.item():
    #                     key_padding_mask = key_padding_mask[:, 0, :, :].reshape(B, H * W)  # [B, H*W]
    #                     return key_padding_mask
    #                 else:
    #                     raise ValueError("Padding mask is not consistent across channels.")

    #         elif key_padding_mask.dim() == 3:
    #             B, H, W = key_padding_mask.shape
    #             key_padding_mask = key_padding_mask.reshape(B, H * W)  # [B, H*W]
    #             return key_padding_mask
    #         elif key_padding_mask.dim() == 2:
    #             return key_padding_mask
    #         else:
    #             raise ValueError("Invalid padding mask dimensions.")
    #     return None

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            key_padding_mask: torch.Tensor = None
            ) -> torch.Tensor:

        # Debug
        # tic = time()
        # print(f"\nQuery Shape: {query.shape}")
        # print(f"Key Shape: {key.shape}")

        B, _, H, W = query.shape

        # * 1. Embed input features
        # [B, H*W, embed_dim]
        query, key = self._process_input(query, key)

        # * Assert key_padding_mask is valid
        # key_padding_mask = self._assert_valid_key_padding_mask(key_padding_mask)

        # * 2. Multi-head attention
        # ! We can only enforce the use of the FLASH backend if we are using
        # ! cuda, `key_padding_mask` is None, and mixed precision is enabled,
        # ! otherwise, the code will raise an error.
        # if torch.cuda.is_available() and key_padding_mask is None:
        #     with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        #         attn_output, _ = self.attn(
        #             query=query,  # upsampled feature map flattened
        #             key=key,  # skip connection flattened
        #             value=key,  # skip connection flattened
        #             key_padding_mask=key_padding_mask,
        #             need_weights=False,
        #             is_causal=False
        #             )
        # else:
        #     attn_output, _ = self.attn(
        #         query=query,  # upsampled feature map flattened
        #         key=key,  # skip connection flattened
        #         value=key,  # skip connection flattened
        #         key_padding_mask=key_padding_mask,
        #         need_weights=False,
        #         is_causal=False
        #         )
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            attn_output, _ = self.attn(
                query=query,  # upsampled feature map flattened
                key=key,  # skip connection flattened
                value=key,  # skip connection flattened
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=False
                )

        # Note: `needs_weight` defaults to `True`, but should be set to `False`
        # For best performance when attention weights are not needed.
        # Setting needs_weights to `True` leads to a significant performance
        # degradation.

        # * 3. Dropout
        attn_output = self.dropout(attn_output)

        # * 4. Layer Normalization
        attn_output = self.ln_1(attn_output)

        # * 5. Feed-forward network
        attn_output = self.ffn(attn_output)

        # * 6. Layer Normalization
        attn_output = self.ln_2(attn_output)

        # * Reshape back to [B, C, H, W]
        attn_output = attn_output.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)

        # * Project back to input feature dimensions
        attn_output = self.conv2(attn_output)

        # toc = time()
        # print(f"Time taken: {toc - tic:.4f} seconds\n")

        return attn_output


if __name__ == '__main__':

    # * Instantiate Attention Module
    cross_attn = CrossAttentionBlock(
        channels=3,
        embed_dim=256,
        embed_num_positions=64 * 64,
        num_heads=4,
        attention_dropout=0.1,
        dropout=0.1,
        use_learnable_positional_embeddings=True,
        ffn_hidden_dim=512,
        ).cuda()

    x = torch.randn(1, 3, 64, 64).cuda()  # Query
    y = torch.randn(1, 3, 64, 64).cuda()  # Key

    # * Compile Attention Module
    # print("Compiling Attention Module...")
    # cross_attn = torch.compile(cross_attn)

    attn_output = cross_attn(
        query=x,
        key=y
        )
    print(f"Attention Output Shape: {attn_output.shape}")
    print(f"Attention Output: {attn_output}")
