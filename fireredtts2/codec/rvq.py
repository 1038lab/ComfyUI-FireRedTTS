import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class VectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_project = (
            WNConv1d(
                self.input_dim, self.codebook_dim, kernel_size=1
            )  # (B, D, T) -> (B, D', T)
            if self.input_dim != self.codebook_dim
            else nn.Identity()
        )
        self.out_project = (
            WNConv1d(
                self.codebook_dim, self.input_dim, kernel_size=1
            )  # (B, D', T) -> (B, D, T)
            if self.input_dim != self.codebook_dim
            else nn.Identity()
        )

        # Initialize codebook and EMA buffers
        self.register_buffer(
            "codebook", torch.zeros(codebook_size, codebook_dim).float()
        )  # (codebook_size, D'), ensure fp32
        # Place holder, not used in inference
        self.register_buffer("inited", torch.tensor([True], dtype=torch.bool))  # (1)
        self.register_buffer(
            "cluster_size", torch.zeros(codebook_size).float()
        )  # (codebook_size), ensure fp32
        self.register_buffer(
            "embed_avg", self.codebook.clone().float()
        )  # (codebook_size, D'), ensure fp32

    def decode_code(self, embed_id):  # embed_id: (B, T)
        embed = (
            F.embedding(embed_id, self.codebook).transpose(1, 2).float()
        )  # (B, D', T), ensure fp32
        return embed

    def encode_code(self, z: torch.Tensor):  # z: (B, D, T)
        # logging.info(f"{self.cluster_size = }, {self.codebook = }, {self.embed_avg = }, {self.inited = }")
        z = z.float()  # Ensure fp32
        z_e = self.in_project(z).float()  # (B, D', T), ensure fp32

        # Rearrange for quantization
        encodings = rearrange(z_e, "b d t -> (b t) d").float()  # (B*T, D'), ensure fp32

        # Quantization
        dist = (
            encodings.pow(2).sum(1, keepdim=True)  # (B*T, 1)
            - 2 * encodings @ self.codebook.float().t()  # (B*T, codebook_size)
            + self.codebook.float().pow(2).sum(1, keepdim=True).t()
        )  # (1, codebook_size)

        # dist: (B*T, codebook_size)
        indices = (-dist).max(1)[1]  # (B*T)
        indices = rearrange(indices, "(b t) -> b t", b=z.size(0))  # (B, T)

        # Get quantized vectors
        z_q = self.decode_code(indices).float()  # (B, D', T), ensure fp32

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()  # (B, D', T)
        z_q = self.out_project(z_q).float()  # (B, D, T), ensure fp32

        # z_q: (B, D, T), commit_loss: (B), indices: (B, T), z: (B, D', T)
        return z_q, indices


class ResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,  # Input dimension, unrelated to RVQ
        rvq_dim=None,  # RVQ dimension. If different from input_dim/output_dim, will add input_dim->rvq_dim/rvq_dim->output_dim projection
        output_dim: int = None,  # Output dimension, unrelated to RVQ
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,  # Dimension of each codebook. If different from rvq_dim, will add rvq_dim->codebook_dim and codebook_dim->rvq_dim projections
    ):
        super().__init__()
        self.input_dim = input_dim

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.rvq_dim = rvq_dim

        self.input_proj = (
            WNConv1d(input_dim, rvq_dim, kernel_size=1)
            if input_dim != rvq_dim
            else nn.Identity()
        )
        self.output_proj = (
            WNConv1d(rvq_dim, output_dim, kernel_size=1)
            if rvq_dim != output_dim
            else nn.Identity()
        )

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(
                    input_dim=rvq_dim,
                    codebook_size=self.codebook_size,
                    codebook_dim=codebook_dim,
                )
                for i in range(num_quantizers)
            ]
        )

    def encode_codes(self, z: torch.Tensor):
        z = self.input_proj(z)
        residual = z.clone().float()  # (B, D, T), ensure fp32
        all_indices = []
        # Quantize to tokens
        for i, quantizer in enumerate(self.quantizers):
            # (B, D, T), (B), scalar, (B, T), (B, D', T), ensure fp32
            z_q_i, indices_i = quantizer.encode_code(residual)
            residual = residual - z_q_i
            all_indices.append(indices_i)  # (B, T)
        all_indices = torch.stack(all_indices)  # (N, B, T)
        return all_indices

    def decode_codes(self, codes):  # codes: (nq, B, T)
        """Decode codes from multiple quantizers to embeddings.

        Args:
            codes: Tensor of shape (nq, B, T) containing code indices for each quantizer.

        Returns:
            emb: Tensor of shape (B, D, T) representing the decoded embeddings.
        """
        nq, B, T = codes.shape
        device = codes.device
        emb = torch.zeros(
            B, self.rvq_dim, T, device=device, dtype=torch.float32
        )  # (B, D, T)
        for i, quantizer in enumerate(self.quantizers[:nq]):
            code_i = codes[i]  # (B, T)
            quantized_i = quantizer.decode_code(code_i)  # (B, D', T)
            emb += quantizer.out_project(quantized_i)  # Accumulate quantized embeddings
        emb = self.output_proj(emb)  # (B, D, T), apply output projection
        return emb  # (B, D, T)
