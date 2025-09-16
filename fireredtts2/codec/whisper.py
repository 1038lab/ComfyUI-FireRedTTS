# Extracted from transformers' WhisperModel to simplify package dependency
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from fireredtts2.codec.utils import make_nonpad_mask
from fireredtts2.codec.audio import mel_filter_bank


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Returns sinusoids for positional embedding"""
    if channels % 2 != 0:
        raise ValueError(
            f"Number of channels has to be divisible by 2 for sinusoidal positional embeddings, got {channels} channels."
        )
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class WhisperSdpaAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            attention_mask: Bool mask or float mask. Bool mask, True indicates should attend. Float mask is added to the attention score.
        """
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        # NOTE sdpa needs a 4-dim attention_mask: (b, nh, tq, tv)
        if attention_mask is not None and len(attention_mask.shape) == 3:
            attention_mask = attention_mask.unsqueeze(1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (bsz, nh, l, d)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor = None,
    ):
        """Forward self-attention with kv cache.

        Args:
            hidden_states: shape (b, t, c)
            kv_cache: shape (b, nh, t, c*2)
        """
        bsz, tgt_len, _ = hidden_states.size()

        # shape (b, nh, t, c)
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)
        key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz)

        # unpack cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache.chunk(2, dim=-1)
            key_states = torch.cat([k_cache, key_states], dim=2)
            value_states = torch.cat([v_cache, value_states], dim=2)
        new_kv_cache = torch.cat([key_states, value_states], dim=-1)

        # attention
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
        )  # (bsz, nh, l, d)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, new_kv_cache


class WhisperEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int = None,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        # Attention
        self.self_attn = WhisperSdpaAttention(embed_dim, num_heads, attn_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        # FFN
        ffn_dim = ffn_dim if ffn_dim is not None else embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        # Output norm
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        # Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor = None,
    ):
        """Forward self-attention with kv cache.

        Args:
            hidden_states: shape (b, t, c)
            kv_cache: shape (b, nh, t, c*2)
        """
        # Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, new_kv_cache = self.self_attn.forward_chunk(
            hidden_states, kv_cache
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        return hidden_states, new_kv_cache


class WhisperEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int = None,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        max_positions: int = 1500,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        # Input downsampling
        self.conv1 = nn.Conv1d(in_dim, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        # Fixed positional embedding
        self.max_positions = max_positions
        self.embed_positions = nn.Embedding(self.max_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        # Transformer
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(
                    embed_dim, num_heads, ffn_dim, attn_dropout, dropout
                )
                for _ in range(num_layers)
            ]
        )
        # Output norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        # Init weight
        self.apply(self._init_weights)
        # Init position embedding
        self.embed_positions.weight.copy_(sinusoids(*self.embed_positions.weight.shape))

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_length: torch.Tensor,
        apply_position: bool = True,
    ):
        # Downsampling
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.gelu(self.conv1(hidden_states))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_length = hidden_length // 2  # from 100Hz -> 50Hz
        # Pos encoding
        if apply_position:
            pos_embed = self.embed_positions(
                torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
            )
            hidden_states = hidden_states + pos_embed
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        # Transformer
        attention_mask = make_nonpad_mask(hidden_length).unsqueeze(1)  # (b, 1, t)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states, hidden_length

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class WhisperMelExtractor(nn.Module):
    def __init__(
        self,
        num_mels: int = 128,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 400,
        fmin: float = 0,
        fmax: float = 8000,
        padding_value=0.0,
    ):
        super().__init__()
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.padding_value = padding_value
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=(1 + n_fft // 2),
            num_mel_filters=num_mels,
            min_frequency=fmin,
            max_frequency=fmax,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def extract_fbank(self, audio: torch.Tensor):
        """
        Args:
            audio: batched audio of shape (b, t)
        """
        device = audio.device  # compute on cuda if input is on cuda
        # Mel
        window = torch.hann_window(self.n_fft).to(device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32).to(device)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # Norm
        max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def __call__(self, audio16k: torch.Tensor, audio16k_length: torch.Tensor):
        mel = self.extract_fbank(audio16k).transpose(1, 2)
        mel_length = audio16k_length // self.hop_length
        # mel: (b, t, c=128)
        return mel, mel_length


# Pretrained encoder from whisper-large-v3
class PretrainedWhisperEncoder(WhisperEncoder):
    @classmethod
    def from_pretrained(cls, pretrained_path: str = None):
        encoder = cls(
            in_dim=128,
            embed_dim=1280,
            num_layers=32,
            num_heads=20,
            ffn_dim=5120,
            attn_dropout=0.0,
            max_positions=1500,
        )
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            encoder.load_state_dict(ckpt)
        encoder.eval()
        # Disable grad
        for p in encoder.parameters():
            p.requires_grad_(False)
        # Add Mel extractor
        encoder.feature_extractor = WhisperMelExtractor(
            num_mels=128,
            sampling_rate=16000,
            hop_length=160,
            n_fft=400,
            fmin=0,
            fmax=8000,
        )
        return encoder

    @torch.inference_mode()
    def forward(self, audio16k: torch.Tensor, audio16k_length: torch.Tensor):
        # Extract mel
        mel, mel_length = self.feature_extractor(audio16k, audio16k_length)
        # Forward model
        semantic_feats, semantic_length = super().forward(
            mel, mel_length, apply_position=True
        )
        return semantic_feats, semantic_length


class WhisperAcousticEncoder(WhisperEncoder):
    def __init__(
        self,
        # Mel extraction params
        num_mels: int = 128,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 400,
        fmin: float = 0.0,
        fmax: float = 8000,
        # Encoder params
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = None,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        max_positions: int = 1500,  # 50Hz * 30s
    ):
        super().__init__(
            in_dim=num_mels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            attn_dropout=attn_dropout,
            dropout=dropout,
            max_positions=max_positions,
        )
        self.feature_extractor = WhisperMelExtractor(
            num_mels=num_mels,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            fmin=fmin,
            fmax=fmax,
        )

    def forward(self, audio16k: torch.Tensor, audio16k_length: torch.Tensor):
        # Extract mel
        with torch.no_grad():
            mel, mel_length = self.feature_extractor(audio16k, audio16k_length)
        # Forward model
        hidden_states, hidden_length = super().forward(
            mel, mel_length, apply_position=True
        )
        return hidden_states, hidden_length
