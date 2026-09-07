# Based on Pathway's public toy BDH implementation.
# Modified for educational state inspection.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
            * self.freqs
        )
        QR = self.rope(r_phases, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _round_list(values, digits=5):
        return [round(float(v), digits) for v in values]

    def _capture_state(self, level, x_latent, x_sparse, y_sparse, xy_sparse):
        """Return compact, JSON-friendly measurements from real BDH tensors."""
        with torch.no_grad():
            pre = x_latent.detach().float()
            post = x_sparse.detach().float()
            y = y_sparse.detach().float()
            xy = xy_sparse.detach().float()

            pre_nonpositive = (pre <= 0).float().mean().item() * 100
            post_zero = (post == 0).float().mean().item() * 100
            x_active = (post > 0).float().mean().item() * 100
            y_active = (y > 0).float().mean().item() * 100
            xy_active = (xy > 0).float().mean().item() * 100

            head_activity = (
                (post > 0).float().mean(dim=(0, 2, 3)).cpu().tolist()
            )
            head_activity = [v * 100 for v in head_activity]

            token_activity = (
                (post > 0).float().mean(dim=(0, 1, 3)).cpu().tolist()
            )
            token_activity = [v * 100 for v in token_activity]

            # Strongest last-token post-ReLU units across all heads.
            last_post = post[0, :, -1, :].reshape(-1)
            k = min(48, last_post.numel())
            top_vals, top_idx = torch.topk(last_post, k=k)

            # Small real pre/post-ReLU example for the UI.
            last_pre_head0 = pre[0, 0, -1, :]
            sample_k = min(16, last_pre_head0.numel())
            _, sample_idx = torch.topk(last_pre_head0.abs(), k=sample_k)
            sample_pre = last_pre_head0[sample_idx]
            sample_post = F.relu(sample_pre)

            return {
                "layer": level + 1,
                "x_active_percent": round(x_active, 4),
                "y_active_percent": round(y_active, 4),
                "xy_active_percent": round(xy_active, 4),
                "pre_relu_nonpositive_percent": round(pre_nonpositive, 4),
                "post_relu_zero_percent": round(post_zero, 4),
                "relu_match_error_percent": round(abs(pre_nonpositive - post_zero), 8),
                "mean_activation": round(post.mean().item(), 6),
                "max_activation": round(post.max().item(), 6),
                "head_activity_percent": self._round_list(head_activity, 4),
                "token_activity_percent": self._round_list(token_activity, 4),
                "last_token_top_units": [
                    {"unit": int(i), "activation": round(float(v), 6)}
                    for i, v in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist())
                ],
                "relu_examples": [
                    {
                        "unit": int(i),
                        "pre": round(float(a), 5),
                        "post": round(float(b), 5),
                    }
                    for i, a, b in zip(
                        sample_idx.cpu().tolist(),
                        sample_pre.cpu().tolist(),
                        sample_post.cpu().tolist(),
                    )
                ],
            }

    def forward(self, idx, targets=None, return_states=False):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)
        states = []

        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse

            if return_states:
                states.append(
                    self._capture_state(
                        level, x_latent, x_sparse, y_sparse, xy_sparse
                    )
                )

            xy_sparse = self.drop(xy_sparse)
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_states:
            return logits, loss, states
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
