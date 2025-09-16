import math
import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask  # (b, t)


def make_nonpad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    return ~make_pad_mask(lengths, max_len)


def make_block_causal_mask(
    lengths: torch.Tensor, max_len: int = 0, chunk_size: int = 4
) -> torch.Tensor:
    mask = make_nonpad_mask(lengths, max_len)  # (b, t)
    attn_mask = torch.logical_and(mask.unsqueeze(1), mask.unsqueeze(2))  # (b, t, t)

    num_blocks = math.ceil(attn_mask.shape[1] / chunk_size)
    block_mask = torch.block_diag(
        *[torch.ones(chunk_size, chunk_size) for _ in range(num_blocks)]
    )
    block_mask = block_mask[: attn_mask.shape[1], : attn_mask.shape[1]].to(
        attn_mask
    )  # (t, t)

    diag_mask = attn_mask.new_full(
        (1, attn_mask.shape[1], attn_mask.shape[2]), fill_value=True
    ).tril()  # (1, t, t)
    diag_mask = diag_mask.logical_or(block_mask)
    attn_mask = attn_mask.logical_and(diag_mask)
    return attn_mask
