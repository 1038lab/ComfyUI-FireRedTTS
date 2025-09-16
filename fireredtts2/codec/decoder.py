import torch
import torch.nn as nn
import torch.nn.functional as F
from fireredtts2.codec.whisper import WhisperEncoderLayer
from fireredtts2.codec.utils import make_nonpad_mask, make_block_causal_mask


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.block1 = nn.Sequential(
            nn.GroupNorm(
                num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
            ),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(
                num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: shape (b, c, t)
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


# A causal variant of Conv1d
class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.causal_padding)
        x = super(CausalConv1d, self).forward(x)
        return x

    def forward_chunk(self, x: torch.Tensor, cnn_cache: torch.Tensor = None):
        if cnn_cache is None:
            cnn_cache = x.new_zeros(
                (x.shape[0], self.in_channels, self.causal_padding[0])
            )
        x = torch.cat([cnn_cache, x], dim=2)
        new_cnn_cache = x[..., -self.causal_padding[0] :]
        x = super(CausalConv1d, self).forward(x)
        return x, new_cnn_cache


# A causal variant of ResnetBlock
class CausalResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.block1 = nn.Sequential(
            Transpose(1, 2),
            nn.LayerNorm(in_channels),
            Transpose(1, 2),
            nn.SiLU(),
            CausalConv1d(in_channels, out_channels, kernel_size=3),
        )

        self.block2 = nn.Sequential(
            Transpose(1, 2),
            nn.LayerNorm(out_channels),
            Transpose(1, 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size=3),
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: shape (b, c, t)
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h

    def forward_chunk(self, x: torch.Tensor, cache: torch.Tensor = None):
        """
        Args:
            x: shape (b, c, t)
            cache: shape (b, c_in+c_out, t=2)
        """
        cache1, cache2 = (
            (None, None)
            if cache is None
            else cache.split((self.in_channels, self.out_channels), dim=1)
        )
        h = x
        # block1
        h = self.block1[:4](h)
        h, new_cache1 = self.block1[4].forward_chunk(h, cache1)
        # block2
        h = self.block2[:5](h)
        h, new_cache2 = self.block2[5].forward_chunk(h, cache2)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        new_cache = torch.cat([new_cache1, new_cache2], dim=1)
        return x + h, new_cache


# Nonstreaming Vocos backbone based on Transformer layers
class VocosBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3)
        self.prior_net = nn.Sequential(
            ResnetBlock(embed_dim, embed_dim, dropout=dropout),
            ResnetBlock(embed_dim, embed_dim, dropout=dropout),
        )
        self.transformers = nn.ModuleList(
            [WhisperEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.post_net = nn.Sequential(
            ResnetBlock(embed_dim, embed_dim, dropout=dropout),
            ResnetBlock(embed_dim, embed_dim, dropout=dropout),
        )
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ):
        """
        Args:
            x: shape (b, t, c)
            x_lens: shape (b,)
        """
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.prior_net(x)
        x = x.transpose(1, 2)

        attention_mask = make_nonpad_mask(x_lens).unsqueeze(1)  # (b, 1, t)
        # NOTE(sfy): I think positional embedding is unnecessary
        for layer in self.transformers:
            x = layer(x, attention_mask)
        x = x.transpose(1, 2)
        x = self.post_net(x)
        x = x.transpose(1, 2)
        x = self.final_norm(x)
        return x


# Streaming Vocos backbone based on Transformer layers
class CausalVocosBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = CausalConv1d(embed_dim, embed_dim, kernel_size=7)
        self.prior_net = nn.Sequential(
            CausalResnetBlock(embed_dim, embed_dim, dropout=dropout),
            CausalResnetBlock(embed_dim, embed_dim, dropout=dropout),
        )
        self.transformers = nn.ModuleList(
            [WhisperEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.post_net = nn.Sequential(
            CausalResnetBlock(embed_dim, embed_dim, dropout=dropout),
            CausalResnetBlock(embed_dim, embed_dim, dropout=dropout),
        )
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ):
        """
        Args:
            x: shape (b, t, c)
            x_lens: shape (b,)
        """
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.prior_net(x)
        x = x.transpose(1, 2)

        # NOTE(sfy): We have no padding in training, so safe for sdpa attention, no Nan.
        # Also, 1 token(12.5Hz) -> 4 latents(50Hz) -> 8 latents(100Hz),
        # so we design a 8 block causal attention mask instead of fully causal to improve performance
        attention_mask = make_block_causal_mask(x_lens, chunk_size=8)
        for layer in self.transformers:
            x = layer(x, attention_mask)

        x = x.transpose(1, 2)
        x = self.post_net(x)
        x = x.transpose(1, 2)
        x = self.final_norm(x)
        return x

    def forward_chunk(
        self,
        x: torch.Tensor,
        conv_cache1: torch.Tensor = None,
        conv_cache2: torch.Tensor = None,
        kv_cache: torch.Tensor = None,
    ):
        # Unpack cache
        cache1 = conv_cache1
        cache2, cache3, cache4, cache5 = (
            (None, None, None, None)
            if conv_cache2 is None
            else conv_cache2.chunk(4, dim=1)
        )

        # cache1: shape (b, c=embed_dim, t=6)
        x = x.transpose(1, 2)
        x, new_cache1 = self.in_proj.forward_chunk(x, cache1)
        # cache2: shape (b, c=embed_dim*2, t=2)
        x, new_cache2 = self.prior_net[0].forward_chunk(x, cache2)
        # cache3: shape (b, c=embed_dim*2, t=2)
        x, new_cache3 = self.prior_net[1].forward_chunk(x, cache3)
        x = x.transpose(1, 2)

        # k,v-cache: shape (b, nlayer, nh, t, c*2)
        new_kv_cache = []
        for idx, layer in enumerate(self.transformers):
            kv_cache_i = None if kv_cache is None else kv_cache[:, idx]
            x, new_kv_cache_i = layer.forward_chunk(x, kv_cache=kv_cache_i)
            new_kv_cache.append(new_kv_cache_i)
        new_kv_cache = torch.stack(new_kv_cache, dim=1)

        x = x.transpose(1, 2)
        # cache4: shape (b, c=embed_dim*2, t=2)
        x, new_cache4 = self.post_net[0].forward_chunk(x, cache4)
        # cache5: shape (b, c=embed_dim*2, t=2)
        x, new_cache5 = self.post_net[1].forward_chunk(x, cache5)
        x = x.transpose(1, 2)
        x = self.final_norm(x)

        new_conv_cache1 = new_cache1
        new_conv_cache2 = torch.cat(
            [new_cache2, new_cache3, new_cache4, new_cache5], dim=1
        )
        return x, new_conv_cache1, new_conv_cache2, new_kv_cache


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"
    ):
        super().__init__()
        assert padding in ["center", "same"], "Padding must be 'center' or 'same'."
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y

    def forward_chunk(
        self, spec: torch.Tensor, cache: torch.Tensor = None, last_chunk: bool = False
    ):
        """Forward only one frame.

        Args:
            spec: shape (B, N, T=chunk_size)
            cache: previous chunk's last ifft frame, shape (B, N, T=3)
            last_chunk: if last_chunk, will not trim the last (win-hop) segment
        Returns:
            y: shape (B, T=effective_length)
        """
        assert self.padding == "same", "Padding must be same."
        assert (
            self.win_length % self.hop_length == 0
        ), f"{self.win_length} {self.hop_length}"
        pad = (self.win_length - self.hop_length) // 2

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]  # (B, N, T=chunk_size)

        # Append previous cache
        if cache is not None:
            ifft = torch.cat([cache, ifft], dim=-1)
        new_cache_t = self.win_length // self.hop_length - 1
        new_cache = ifft[..., -new_cache_t:]

        # Overlap and Add
        output_size = (ifft.shape[-1] - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]

        # Window envelope
        window_sq = (
            self.window.square().expand(1, ifft.shape[-1], -1).transpose(1, 2)
        )  # (B=1, N, T)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()

        # Normalize
        # assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        # Only take effective part
        if cache is None:
            y = y[:, pad:]
        else:
            y = y[:, (self.win_length - self.hop_length) :]
        if last_chunk:
            y = y[:, :-pad]
        else:
            y = y[:, : -(self.win_length - self.hop_length)]
        return y, new_cache


class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        self.hop_length = hop_length
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        audio_length = x_len * self.hop_length
        return audio, audio_length

    def forward_chunk(
        self, x: torch.Tensor, cache: torch.Tensor = None, last_chunk: bool = False
    ):
        """ISTFTHead can be adapted in streaming inference without retraining.

        Args:
            x: shape (B, T, C)
            cache: shape (B, N, T=3), istft cache
        Returns:
            audio: shape (B, t)
        """
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)  # (B, C, T)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)  # (B, C, T)
        audio, new_cache = self.istft.forward_chunk(S, cache, last_chunk)
        return audio, new_cache


# UpsampleConv(50->100Hz) + VocosBackbone + ISTFTHead
class AcousticDecoder(nn.Module):
    def __init__(
        self,
        # Transformer
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
        # iSTFT
        hop_length: int = 240,
        # Causal
        causal: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hop_length = hop_length
        self.causal = causal

        # Output upsample
        self.upsample_conv = nn.Sequential(
            nn.ConvTranspose1d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=0,  # Do not fill input side
                output_padding=0,  # Can be adjusted to precisely control length
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                stride=1,
                padding=0,  # Do not fill input side
            ),
            nn.GELU(),
        )
        self.backbone = (
            CausalVocosBackbone(embed_dim, num_layers, num_heads, dropout)
            if causal
            else VocosBackbone(embed_dim, num_layers, num_heads, dropout)
        )
        self.isift = ISTFTHead(embed_dim, hop_length * 4, hop_length, padding="same")
        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        """
        Args:
            x: shape (b, t, c)
            x_lens: shape (b,)
        """
        # Upsample
        target_length = x.shape[1] * 2
        x = x.transpose(1, 2)
        x = self.upsample_conv(x)
        x = x.transpose(1, 2)
        # NOTE strict upsampling, trim the last 3 elements
        x = x[:, :target_length]
        x_lens = x_lens * 2
        # Backbone
        x = self.backbone(x, x_lens)
        # iSTFT
        y, y_lens = self.isift(x, x_lens)
        return y, y_lens

    def forward_upsample_conv_chunk(self, x: torch.Tensor, cache: torch.Tensor = None):
        """Stream forward upsample_conv module with previous block cache.

        Args:
            x: shape (B, C, T)
            cache: shape (B, C, 3), where 3 denotes 1 history state for 1st conv and 2 for the rest conv.
        """
        # Unpack cache
        cache1, cache2 = (
            (None, None) if cache is None else torch.split(cache, [1, 2], dim=2)
        )
        # 1st conv cache
        if cache1 is not None:
            x = torch.cat([cache1, x], dim=2)
        new_cache1 = x[..., -1:]
        # 1st conv
        x = self.upsample_conv[0](x)[..., :-1]  # remove extra 1 frame
        if cache1 is not None:
            x = x[..., 2:]  # remove cache1 part
        x = self.upsample_conv[1](x)
        # 2nd conv cache
        if cache2 is not None:
            x = torch.cat([cache2, x], dim=2)
        new_cache2 = x[..., -2:]
        # 2nd conv
        x = self.upsample_conv[2](x)[..., :-2]  # remove extra 2 frame
        if cache2 is not None:
            x = x[..., 2:]  # remove cache2 part
        x = self.upsample_conv[3](x)

        new_cache = torch.cat([new_cache1, new_cache2], dim=2)
        return x, new_cache

    def forward_chunk(
        self,
        x: torch.Tensor,
        # Upsample conv cache
        up_conv_cache: torch.Tensor = None,
        # Backbone conv cache
        bb_conv_cache1: torch.Tensor = None,
        bb_conv_cache2: torch.Tensor = None,
        # Backbone attention cache
        bb_kv_cache: torch.Tensor = None,
        # iSTFT cache
        is_cache: torch.Tensor = None,
        last_chunk: bool = False,
    ):
        """
        Args:
            x: input sequence at 50Hz, length should be multiples of 4
        """
        assert (
            self.causal
        ), "Only AcousticDecoder with causal=True supports forward_chunk method."

        x = x.transpose(1, 2)
        x, new_up_conv_cache = self.forward_upsample_conv_chunk(x, up_conv_cache)
        x = x.transpose(1, 2)
        # Backbone
        x, new_bb_conv_cache1, new_bb_conv_cache2, new_bb_kv_cache = (
            self.backbone.forward_chunk(
                x,
                bb_conv_cache1,
                bb_conv_cache2,
                bb_kv_cache,
            )
        )
        # iSTFT
        y, new_is_cache = self.isift.forward_chunk(x, is_cache, last_chunk)
        return (
            y,
            new_up_conv_cache,
            new_bb_conv_cache1,
            new_bb_conv_cache2,
            new_bb_kv_cache,
            new_is_cache,
        )
