import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)


def _npu_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out, _ = torch_npu.npu_rms_norm(x, weight, eps)
    return out


def _per_layer_context_kv(
    self,
    context_states: torch.Tensor,
    num_ctx: int,
    L: int,
    nkv: int,
    hd: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dnest per-layer context KV: slice j of context_states is draft layer
    j's own tap, normed by that layer's input_layernorm (its context_kv
    copy in kv_mode='split'/'split_trainable') and projected by its
    context K/V path — the same buffers
    _build_per_layer_context_kv_buffers stacked."""
    hidden = self.config.hidden_size
    expected = L * hidden
    if context_states.shape[-1] != expected:
        raise ValueError(
            f"per_layer context expects [{expected} = {L} layers x {hidden} "
            f"hidden] concatenated hidden states, got "
            f"{context_states.shape[-1]}. Check that the draft config's "
            "aux_hidden_state_layer_ids matches the training taps."
        )
    layered = (
        context_states[:, :expected]
        .view(num_ctx, L, hidden)
        .permute(1, 0, 2)
        .contiguous()
    )
    is_split = self.kv_mode in ("split", "split_trainable")
    normed = torch.empty_like(layered)
    if is_split:
        for j in range(L):
            normed[j] = _npu_rms_norm(
                layered[j], self._context_input_norm_weights[j], self._rms_norm_eps
            )
    else:
        for j in range(L):
            normed[j] = self.layers[j].input_layernorm(layered[j])
    all_kv = torch.bmm(normed, self._per_layer_kv_weight).view(L, num_ctx, 2, nkv, hd)
    all_k = all_kv[:, :, 0].contiguous()  # [L, num_ctx, nkv, hd]
    all_v = all_kv[:, :, 1].contiguous()  # [L, num_ctx, nkv, hd]

    # --- Per-layer RMSNorm K (split norms with the frozen context_kv copies) ---
    all_k_normed = torch.empty_like(all_k)
    if is_split:
        for j in range(L):
            all_k_normed[j] = _npu_rms_norm(
                all_k[j], self._k_norm_weights[j], self._rms_norm_eps
            )
    else:
        for j in range(L):
            all_k_normed[j] = self.layers[j].self_attn.k_norm(all_k[j])
    return all_k, all_v, all_k_normed


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    if getattr(self, "context_mode", "fused") == "per_layer":
        all_k, all_v, all_k_normed = _per_layer_context_kv(
            self, context_states, num_ctx, L, nkv, hd
        )
    else:
        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = self.hidden_norm(context_states)
        all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

        # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            k_norm_layer = self.layers[i].self_attn.k_norm
            all_k_normed[i] = k_norm_layer(all_k[i])

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    per_layer = isinstance(context_slot_mapping, (list, tuple))
    for i in range(L):
        slot_mapping = context_slot_mapping[i] if per_layer else context_slot_mapping
        if slot_mapping is None:
            continue
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            kv_cache,
            slot_mapping,
        )


DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv

_orig_read_mask_embedding = DFlashQwen3ForCausalLM._read_mask_embedding


def _patched_read_mask_embedding(self):
    try:
        return _orig_read_mask_embedding(self)
    except Exception:
        return None


DFlashQwen3ForCausalLM._read_mask_embedding = _patched_read_mask_embedding
