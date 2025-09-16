from torchtune.models.qwen2 import qwen2
from torchtune.modules.transformer import TransformerDecoder


def qwen2_200M() -> TransformerDecoder:
    return qwen2(
        vocab_size=151936,
        num_layers=4,
        num_heads=12,
        num_kv_heads=2,
        embed_dim=1536,
        intermediate_dim=8960,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_500M() -> TransformerDecoder:
    return qwen2(
        vocab_size=151936,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        embed_dim=896,
        intermediate_dim=4864,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_1_5B() -> TransformerDecoder:
    return qwen2(
        vocab_size=151936,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        embed_dim=1536,
        intermediate_dim=8960,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_3B() -> TransformerDecoder:
    return qwen2(
        vocab_size=151936,
        num_layers=36,
        num_heads=16,
        num_kv_heads=2,
        embed_dim=2048,
        intermediate_dim=11008,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_7B() -> TransformerDecoder:
    return qwen2(
        vocab_size=152064,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        embed_dim=3584,
        intermediate_dim=18944,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
    )


FLAVORS = {
    "qwen-200m": qwen2_200M,
    "qwen-500m": qwen2_500M,
    "qwen-1.5b": qwen2_1_5B,
    "qwen-3b": qwen2_3B,
    "qwen-7b": qwen2_7B,
}
