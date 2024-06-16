"""KWT model based on model from https://github.com/ID56/Torch-KWT/blob/main/models/kwt.py"""

from functools import partial

import torch
import torch.fft
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn

# Basically vision transformer, ViT that accepts MFCC + SpecAug. Refer to:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


class PreNorm(nn.Module):
    """
    Pre layer normalization
    """

    def __init__(self, dim, fn):
        """
        Initialises PreNorm module
        :param dim: model dimension
        :param fn: torch module
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward method for PreNorm module
        :param x: input tensor
        :param kwargs: Keyword arguments
        :return:
        """
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    """
    Post layer normalization
    """

    def __init__(self, dim, fn):
        """
        Initialises PostNorm module
        :param dim: model dimension
        :param fn: torch module
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward method for PostNorm module
        :param x: input tensor
        :param kwargs: Keyword arguments
        :return: PostNorm output
        """
        return self.norm(self.fn(x, **kwargs) + x)


class FeedForward(nn.Module):
    """
    Feed forward model
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        """
        Initialises FeedForward module
        :param dim: feedforward dim
        :param hidden_dim: hidden dimension of feedforward layer
        :param dropout: feedforward dropout percentage
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward method for feedforward module
        :param x: input tensor
        :return: FeedForward output
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Attention module
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        """
        Initialises Attention module
        :param dim: transformer dimension
        :param heads: number of attention heads
        :param dim_head: attention head dimension
        :param dropout: attention output dropout
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        Forward method for Attention module
        :param x: input tensor
        :return: Attention module output
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FNetBasicFourierTransform(nn.Module):
    # Class taken from tansformers.models.fnet.modeling_fnet
    def __init__(self, mixDim=(1, 2)):
        super().__init__()
        self._init_fourier_transform(mixDim)

    def _init_fourier_transform(self, mixDim):
        # Dim 1 is patch dimension
        # Dim 2 is embedding dimension
        self.fourier_transform = partial(torch.fft.fftn, dim=mixDim)

    def forward(self, hidden_states):
        # NOTE: We do not use torch.vmap as it is not integrated into PyTorch stable versions.
        # Interested users can modify the code to use vmap from the nightly versions, getting the vmap from here:
        # https://pytorch.org/docs/master/generated/torch.vmap.html. Note that fourier transform methods will need
        # change accordingly.

        outputs = self.fourier_transform(hidden_states).real
        return outputs


class FnetEncoderCustom(nn.Module):
    """
    FnetEncoderCustom to work with FNetFourierTransform and feedForward from KWT model
    """

    def __init__(
        self,
        dim,
        depth,
        mlp_dim,
        pre_norm=True,
        hidden_dropout_prob=0.0,
    ):
        """
        Initializes Transformer model

        Config should contain
        : dim: transformer dimension
        : depth: number of transformer layers or in this case FNet layers
        : mlp_dim: feedForward or MLP dimension
        : pre_norm: specifies whether PreNorm (True) or PostNorm (False) is used
        : hidden_dropout_prob: dropout percentage of FeedForward modules
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        P_Norm = PreNorm if pre_norm else PostNorm

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        P_Norm(dim, FNetBasicFourierTransform()),
                        P_Norm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout=hidden_dropout_prob,
                            ),
                        ),
                    ]
                )
            )
        self.P_Norm = P_Norm

    def forward(self, x):
        """
        Forward method for Transformer model
        :param x: input tensor
        :return: Tuple of model output, hidden states of transformer and attentions from each transformer layer
        """
        hidden_states = []
        attentions = []

        if isinstance(self.P_Norm, PreNorm):
            for attn, ff in self.layers:
                x = attn(x) + x
                attentions.append(x)
                x = ff(x) + x
                hidden_states.append(x)
        else:
            for attn, ff in self.layers:
                x = attn(x)
                attentions.append(x)
                x = ff(x)
                hidden_states.append(x)
        return x, hidden_states, attentions


class Transformer(nn.Module):
    """
    Transformer model
    """

    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, pre_norm=True, dropout=0.0
    ):
        """
        Initialises Transformer model
        :param dim: transformer dimension
        :param depth: number of transformer layers
        :param heads: number of attention heads for each transformer layer
        :param dim_head: dimension of each attention head
        :param mlp_dim: MLP dimension
        :param pre_norm: specifies whether PreNorm (True) or PostNorm (False) is used
        :param dropout: dropout percentage of Attention of FeedForward modules
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        P_Norm = PreNorm if pre_norm else PostNorm

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        P_Norm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        P_Norm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )
        self.P_Norm = P_Norm

    def forward(self, x):
        """
        Forward method for Transformer model
        :param x: input tensor
        :return: Tuple of model output, hidden states of transformer and attentions from each transformer layer
        """
        hidden_states = []
        attentions = []
        if isinstance(self.P_Norm, PreNorm):
            for attn, ff in self.layers:
                x = attn(x) + x
                attentions.append(x)
                x = ff(x) + x
                hidden_states.append(x)
        else:
            for attn, ff in self.layers:
                x = attn(x)
                attentions.append(x)
                x = ff(x)
                hidden_states.append(x)
        return x, hidden_states, attentions

class HybridMixer(nn.Module):
    """
    Transformer model
    """

    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, attention_layers, pre_norm=True, dropout=0.0, 
        hidden_dropout_prob=0.0,
    ):
        """
        Initialises Transformer model
        :param dim: transformer dimension
        :param depth: number of transformer layers
        :param heads: number of attention heads for each transformer layer
        :param dim_head: dimension of each attention head
        :param mlp_dim: MLP dimension
        :param pre_norm: specifies whether PreNorm (True) or PostNorm (False) is used
        :param dropout: dropout percentage of Attention of FeedForward modules
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        P_Norm = PreNorm if pre_norm else PostNorm

        for i in range(depth):
            if i in attention_layers:
                self.layers.append(
                    nn.ModuleList(
                        [
                            P_Norm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            ),
                            P_Norm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        ]
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList(
                        [
                            P_Norm(dim, FNetBasicFourierTransform()),
                            P_Norm(
                                dim,
                                FeedForward(
                                    dim,
                                    mlp_dim,
                                    dropout=hidden_dropout_prob,
                                ),
                            ),
                        ]
                    )
                )
        self.P_Norm = P_Norm

    def forward(self, x):
        """
        Forward method for Transformer model
        :param x: input tensor
        :return: Tuple of model output, hidden states of transformer and attentions from each transformer layer
        """
        hidden_states = []
        attentions = []
        if isinstance(self.P_Norm, PreNorm):
            for attn, ff in self.layers:
                x = attn(x) + x
                attentions.append(x)
                x = ff(x) + x
                hidden_states.append(x)
        else:
            for attn, ff in self.layers:
                x = attn(x)
                attentions.append(x)
                x = ff(x)
                hidden_states.append(x)
        return x, hidden_states, attentions

class KWT(nn.Module):
    """
    KWT model
    """

    def __init__(
        self,
        input_res,
        patch_res,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=1,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        pre_norm=True,
        **kwargs,
    ):
        """
        Initialises KWT model
        :param input_res: input spectrogram size
        :param patch_res: patch size
        :param num_classes: number of keyword classes
        :param dim: transformer dimension
        :param depth: number of transformer layers
        :param heads: number of attention heads
        :param mlp_dim: MLP dimension
        :param pool: specifies whether CLS token or average pooling of transformer model is used for classification
        :param channels: Number of input channels
        :param dim_head: dimension of attention heads
        :param dropout: dropout of transformer attention and feed forward layers
        :param emb_dropout: dropout of embeddings
        :param pre_norm: specifies whether PreNorm (True) or PostNorm (False) is used
        :param kwargs: Keyword arguments
        """
        super().__init__()

        num_patches = int(input_res[0] / patch_res[0] * input_res[1] / patch_res[1])

        patch_dim = channels * patch_res[0] * patch_res[1]
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_res[0],
                p2=patch_res[1],
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.mask_embedding = nn.Parameter(torch.FloatTensor(dim).uniform_())
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, pre_norm, dropout
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        # Create classification head
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(
        self, x, mask=None, output_hidden_states=False, output_attentions=False
    ):
        """
        Forward method of KWT model
        :param x: input tensor
        :param mask: input mask
        :param output_hidden_states: specifies whether hidden states are output
        :param output_attentions: specifies whether attentions are output
        :return: KWT model output, if output_hidden_states and/or output_attentions the classification head is skipped
        """
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # Add cls token embedding
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Mask input
        if mask is not None:
            x[mask] = self.mask_embedding

        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x, hidden_states, attentions = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)

        if any([output_hidden_states, output_attentions]):
            outputs = (
                (self.mlp_head(x), hidden_states)
                if output_hidden_states
                else (self.mlp_head(x),)
            )
            outputs = outputs + (attentions,) if output_attentions else outputs
            return outputs
        return self.mlp_head(x)


class KWTFNet(nn.Module):
    """
    KWT model using Fnet as a replacement for Attention
    """

    def __init__(
        self,
        input_res,
        patch_res,
        num_classes,
        dim,
        depth,
        mlp_dim,
        pool="cls",
        channels=1,
        emb_dropout=0.0,
        pre_norm=True,
        **kwargs,
    ):
        """
        Initialises KWT model
        :param input_res: input spectrogram size
        :param patch_res: patch size
        :param num_classes: number of keyword classes
        :param dim: transformer dimension
        :param depth: number of transformer layers
        :param mlp_dim: MLP dimension
        :param pool: specifies whether CLS token or average pooling of transformer model is used for classification
        :param channels: Number of input channels
        :param dim_head: dimension of attention heads
        :param emb_dropout: dropout of embeddings
        :param pre_norm: specifies whether PreNorm (True) or PostNorm (False) is used
        :param kwargs: Keyword arguments
        """
        super().__init__()

        num_patches = int(input_res[0] / patch_res[0] * input_res[1] / patch_res[1])

        patch_dim = channels * patch_res[0] * patch_res[1]
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_res[0],
                p2=patch_res[1],
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.mask_embedding = nn.Parameter(torch.FloatTensor(dim).uniform_())
        self.transformer = FnetEncoderCustom(dim, depth, mlp_dim, pre_norm=pre_norm)
        self.pool = pool
        self.to_latent = nn.Identity()

        # Create classification head
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(
        self, x, mask=None, output_hidden_states=False, output_attentions=False
    ):
        """
        Forward method of KWT model
        :param x: input tensor
        :param mask: input mask
        :param output_hidden_states: specifies whether hidden states are output
        :param output_attentions: specifies whether attentions are output
        :return: KWT model output, if output_hidden_states and/or output_attentions the classification head is skipped
        """
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # Add cls token embedding
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Mask input
        if mask is not None:
            x[mask] = self.mask_embedding

        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x, hidden_states, attentions = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)

        if any([output_hidden_states, output_attentions]):
            outputs = (
                (self.mlp_head(x), hidden_states)
                if output_hidden_states
                else (self.mlp_head(x),)
            )
            outputs = outputs + (attentions,) if output_attentions else outputs
            return outputs
        return self.mlp_head(x)

class HKWT(nn.Module):
    """
    KWT model
    """

    def __init__(
        self,
        input_res,
        patch_res,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        attention_layers,
        pool="cls",
        channels=1,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        pre_norm=True,
        **kwargs,
    ):
        """
        Initialises KWT model
        :param input_res: input spectrogram size
        :param patch_res: patch size
        :param num_classes: number of keyword classes
        :param dim: transformer dimension
        :param depth: number of transformer layers
        :param heads: number of attention heads
        :param mlp_dim: MLP dimension
        :param pool: specifies whether CLS token or average pooling of transformer model is used for classification
        :param channels: Number of input channels
        :param dim_head: dimension of attention heads
        :param dropout: dropout of transformer attention and feed forward layers
        :param emb_dropout: dropout of embeddings
        :param pre_norm: specifies whether PreNorm (True) or PostNorm (False) is used
        :param kwargs: Keyword arguments
        """
        super().__init__()

        num_patches = int(input_res[0] / patch_res[0] * input_res[1] / patch_res[1])

        patch_dim = channels * patch_res[0] * patch_res[1]
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_res[0],
                p2=patch_res[1],
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.mask_embedding = nn.Parameter(torch.FloatTensor(dim).uniform_())
        self.transformer = HybridMixer(
            dim, depth, heads, dim_head, mlp_dim, attention_layers, pre_norm, dropout
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        # Create classification head
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(
        self, x, mask=None, output_hidden_states=False, output_attentions=False
    ):
        """
        Forward method of KWT model
        :param x: input tensor
        :param mask: input mask
        :param output_hidden_states: specifies whether hidden states are output
        :param output_attentions: specifies whether attentions are output
        :return: KWT model output, if output_hidden_states and/or output_attentions the classification head is skipped
        """
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # Add cls token embedding
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Mask input
        if mask is not None:
            x[mask] = self.mask_embedding

        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x, hidden_states, attentions = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)

        if any([output_hidden_states, output_attentions]):
            outputs = (
                (self.mlp_head(x), hidden_states)
                if output_hidden_states
                else (self.mlp_head(x),)
            )
            outputs = outputs + (attentions,) if output_attentions else outputs
            return outputs
        return self.mlp_head(x)