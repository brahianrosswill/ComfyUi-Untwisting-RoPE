from __future__ import annotations

from typing import Any, List

ARCHITECTURE = "anima"
DISPLAY_NAME = "Anima"

# ComfyUI BaseModel stores the selected supported-model instance on
# model.model_config. Anima is declared as comfy.supported_models.Anima.
COMFY_MODEL_CONFIG_CLASS = "Anima"
DIFFUSION_ATTR_PATHS = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)


def matches_model(model_info: dict[str, Any]) -> bool:
    """Select Anima only from ComfyUI's explicit MODEL metadata."""
    return str(model_info.get("model_config_class", "")) == COMFY_MODEL_CONFIG_CLASS


def is_model_identity(model_info: dict[str, Any]) -> bool:
    """Backward-compatible alias for older callers."""
    return matches_model(model_info)


def _get_attr_path(root: Any, attr_path: str) -> tuple[Any, bool]:
    obj = root
    for part in attr_path.split("."):
        if obj is None or not hasattr(obj, part):
            return None, False
        try:
            obj = getattr(obj, part)
        except Exception:
            return None, False
    return obj, True


def find_diffusion_model(model_patcher: Any) -> Any:
    """Return ComfyUI BaseModel.diffusion_model after metadata selected this adapter."""
    for path in DIFFUSION_ATTR_PATHS:
        obj, ok = _get_attr_path(model_patcher, path)
        if ok and obj is not None:
            return obj
    raise RuntimeError("Could not find ComfyUI BaseModel.diffusion_model for Anima.")


def is_self_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """Anima self-attention modules are named blocks.N.self_attn."""
    parts = str(name).split(".")
    if len(parts) != 3:
        return False
    if parts[0] != "blocks" or parts[2] != "self_attn":
        return False
    try:
        idx = int(parts[1])
    except Exception:
        return False
    return int(min_layer) <= idx <= int(max_layer)


def block_index_from_name(name: str) -> int:
    try:
        parts = str(name).split(".")
        if len(parts) >= 2 and parts[0] == "blocks":
            return int(parts[1])
    except Exception:
        pass
    return -1


def is_attention_module(module: Any) -> bool:
    required_attrs = (
        "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "v_norm",
        "output_proj", "output_dropout", "attn_op", "n_heads", "head_dim",
        "compute_qkv", "compute_attention", "forward", "is_selfattn",
    )
    return all(hasattr(module, attr) for attr in required_attrs)


def axes_dims_from_head_dim(head_dim: int) -> List[int]:
    """ComfyUI Cosmos VideoRopePosition3DEmb uses [temporal, height, width] chunks."""
    hd = max(0, int(head_dim))
    dim_h = (hd // 6) * 2
    dim_w = dim_h
    dim_t = hd - 2 * dim_h
    axes = [dim_t, dim_h, dim_w]
    if sum(axes) != hd or any(v <= 0 for v in axes):
        return [hd]
    return axes


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    """Architecture-specific cfg fields merged into the main runtime cfg."""
    cfg: dict[str, Any] = {"architecture": ARCHITECTURE}
    if dm is not None:
        try:
            cfg["axes_dims"] = axes_dims_from_head_dim(
                int(getattr(getattr(dm, "blocks")[0].self_attn, "head_dim"))
            )
        except Exception:
            cfg["axes_dims"] = []
    # Anima self-attention receives only latent/image tokens as [B, T*H*W, D].
    # There is no patchify hook to populate target_real_range, so the attention
    # patch clamps this intentionally huge range to the sequence length.
    cfg["target_qk_adain_ranges"] = [(0, 2 ** 31 - 1)]
    return cfg

# Adapter-owned optional reference-conditioning preprocessing and attention patch.
# Keeping this here is what lets the top-level __init__.py stay model-neutral.

import traceback
import types
import torch
from typing import Optional, Tuple


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y", "t")
    return bool(value)


def _first_tensor_in_conditioning_entry(entry: Any) -> Tuple[Optional[torch.Tensor], dict[str, Any]]:
    meta: dict[str, Any] = {}
    if torch.is_tensor(entry):
        return entry, meta
    if isinstance(entry, dict):
        meta.update(entry)
        for key in ("c_crossattn", "crossattn", "conditioning", "cond", "context", "cap_feats"):
            value = entry.get(key)
            if torch.is_tensor(value):
                return value, meta
        for value in entry.values():
            if torch.is_tensor(value) and value.ndim >= 2:
                return value, meta
        return None, meta
    if isinstance(entry, (list, tuple)):
        cond: Optional[torch.Tensor] = None
        for item in entry:
            if torch.is_tensor(item) and cond is None:
                cond = item
            elif isinstance(item, dict):
                meta.update(item)
        if cond is not None:
            return cond, meta
        for item in entry:
            cond, nested_meta = _first_tensor_in_conditioning_entry(item)
            if nested_meta:
                meta.update(nested_meta)
            if cond is not None:
                return cond, nested_meta
    return None, meta


def _extract_reference_conditioning(ref_conditioning: Any) -> Tuple[Optional[torch.Tensor], dict[str, Any]]:
    if ref_conditioning is None:
        return None, {}
    if torch.is_tensor(ref_conditioning) or isinstance(ref_conditioning, dict):
        return _first_tensor_in_conditioning_entry(ref_conditioning)
    if isinstance(ref_conditioning, (list, tuple)):
        merged_meta: dict[str, Any] = {}
        for entry in ref_conditioning:
            cond, meta = _first_tensor_in_conditioning_entry(entry)
            if meta:
                merged_meta.update(meta)
            if cond is not None:
                return cond, merged_meta
        return None, merged_meta
    return None, {}


def _tensor_batch_ids_like(value: Any, device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    try:
        if torch.is_tensor(value):
            ids = value.detach().to(device=device)
        else:
            ids = torch.as_tensor(value, device=device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        elif ids.ndim > 2:
            ids = ids.reshape(ids.shape[0], -1)
        return ids.long()
    except Exception:
        return None


def _tensor_t5_weights_like(value: Any, like: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    try:
        if torch.is_tensor(value):
            w = value.detach().to(device=like.device, dtype=like.dtype)
        else:
            w = torch.as_tensor(value, device=like.device, dtype=like.dtype)
        if w.ndim == 1:
            w = w.unsqueeze(0).unsqueeze(-1)
        elif w.ndim == 2:
            w = w.unsqueeze(-1)
        return w
    except Exception:
        return None


def prepare_reference_conditioning(
    ref_conditioning: Any,
    dm: Any,
    device,
    dtype,
    stats: Any = None,
    label: str = "",
    helpers: dict[str, Any] | None = None,
) -> Tuple[Any, str]:
    prefix = (helpers or {}).get("prefix", "[UntwistingRoPE]")
    if ref_conditioning is None:
        return ref_conditioning, "none"
    if dm is None or not hasattr(dm, "preprocess_text_embeds"):
        return ref_conditioning, "not-applicable"

    ref_cond, ref_meta = _extract_reference_conditioning(ref_conditioning)
    if ref_cond is None:
        return ref_conditioning, "no-reference-tensor"

    try:
        ref_shape_before = tuple(ref_cond.shape)
        ref_cond_b = ref_cond.detach()
        if ref_cond_b.ndim == 2:
            ref_cond_b = ref_cond_b.unsqueeze(0)

        # Already in the final cross-attention shape. Keep it unchanged.
        if ref_cond_b.ndim >= 3 and int(ref_cond_b.shape[1]) == 512:
            return ref_conditioning, f"already-final-{tuple(ref_cond_b.shape)}"

        t5xxl_ids = ref_meta.get("t5xxl_ids", None)
        if t5xxl_ids is None:
            msg = f"raw-reference-no-t5xxl_ids-shape={ref_shape_before}"
            print(
                f"{prefix} ⚠ reference conditioning is not 512 tokens and has no t5xxl_ids; "
                f"cannot run preprocess_text_embeds safely. shape={ref_shape_before}"
            )
            return ref_conditioning, msg

        ref_cond_b = ref_cond_b.to(device=device, dtype=dtype)
        ids = _tensor_batch_ids_like(t5xxl_ids, device=device)
        if ids is None:
            print(f"{prefix} ⚠ reference conditioning has t5xxl_ids, but they could not be converted to a tensor.")
            return ref_conditioning, f"t5xxl_ids-invalid-shape={ref_shape_before}"

        weights = _tensor_t5_weights_like(ref_meta.get("t5xxl_weights", None), ref_cond_b)

        with torch.inference_mode():
            processed = dm.preprocess_text_embeds(ref_cond_b, ids, t5xxl_weights=weights)
        processed = processed.to(device=device, dtype=dtype).detach()

        out_meta = dict(ref_meta)
        out_meta["num_tokens"] = int(processed.shape[1]) if processed.ndim >= 2 else 0
        out_meta["untwist_adapter_preprocessed"] = True
        out = [[processed, out_meta]]

        status = f"preprocessed-ref-conditioning {ref_shape_before}->{tuple(processed.shape)}"
        return out, status
    except Exception as exc:
        print(f"{prefix} ⚠ reference conditioning preprocess failed; using original conditioning: {exc}")
        if stats is not None and (_coerce_bool(getattr(stats, "verbose", False)) or _coerce_bool(getattr(stats, "rf_verbose", False))):
            traceback.print_exc()
        return ref_conditioning, f"preprocess-failed:{exc}"


def patch_attention_modules(dm: Any, stats: Any, helpers: dict[str, Any] | None = None):
    helpers = helpers or {}
    prefix = helpers.get("prefix", "[UntwistingRoPE]")
    config_key = helpers.get("config_key", "untwisting_rope")
    lerp = helpers["lerp"]
    cross_batch_adain_qk = helpers["cross_batch_adain_qk"]
    build_frequency_scale_vector = helpers["build_frequency_scale_vector"]

    matched = installed = restored = 0
    patched_names: list[str] = []

    for name, module in dm.named_modules():
        if not is_self_attention_name(name, 0, 999):
            continue
        if not is_attention_module(module):
            continue
        if not bool(getattr(module, "is_selfattn", False)):
            continue

        matched += 1
        patched_names.append(name)
        block_idx_for_module = block_index_from_name(name)

        if hasattr(module, "_untwist_orig_adapter_forward"):
            module.forward = module._untwist_orig_adapter_forward
            restored += 1
        else:
            module._untwist_orig_adapter_forward = module.forward
        original_forward = module._untwist_orig_adapter_forward

        def make_forward(orig, module_name, module_block_idx):
            def patched_forward(self, x, context=None, rope_emb=None, transformer_options={}):
                cfg = (
                    transformer_options.get(config_key)
                    if isinstance(transformer_options, dict) else None
                )
                if not cfg or not cfg.get("enabled"):
                    return orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
                if context is not None or not bool(getattr(self, "is_selfattn", False)):
                    return orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)

                block_idx = int(module_block_idx)
                active_blocks = cfg.get("active_blocks", None)
                if active_blocks is not None and len(active_blocks) > 0 and block_idx not in active_blocks:
                    return orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)

                target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                if target_bsz <= 0 or not torch.is_tensor(x) or x.ndim != 3:
                    return orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)

                bsz, seqlen, _ = x.shape
                if bsz < target_bsz * 2:
                    return orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)

                try:
                    if hasattr(stats, "adapter_attn_calls"):
                        stats.adapter_attn_calls += 1
                    q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)

                    progress = float(cfg.get("progress", 0.0))
                    high_scale = lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
                    low_scale = lerp(cfg["low_scale_start"], cfg["low_scale_end"], progress)
                    beta = float(cfg.get("beta", 2.0))

                    if cfg.get("apply_adain") and float(cfg.get("adain_strength", 0)) > 0:
                        q, k = q.clone(), k.clone()
                        q, k = cross_batch_adain_qk(q, k, cfg, target_bsz, float(cfg["adain_strength"]))

                    axes_dims = cfg.get("axes_dims") or axes_dims_from_head_dim(int(self.head_dim))
                    scale_vec = build_frequency_scale_vector(
                        int(self.head_dim), axes_dims,
                        high_scale, low_scale, beta,
                        k.device, k.dtype,
                        runtime_cfg=cfg,
                    ).view(1, 1, 1, int(self.head_dim))

                    q_t = q[:target_bsz]
                    k_t = torch.cat([k[:target_bsz], k[target_bsz:target_bsz * 2] * scale_vec], dim=1)
                    v_t = torch.cat([v[:target_bsz], v[target_bsz:target_bsz * 2]], dim=1)
                    out_t = self.attn_op(q_t, k_t, v_t, transformer_options=transformer_options)

                    q_r = q[target_bsz:target_bsz * 2]
                    k_r = k[target_bsz:target_bsz * 2]
                    v_r = v[target_bsz:target_bsz * 2]
                    out_r = self.attn_op(q_r, k_r, v_r, transformer_options=transformer_options)

                    outs = [out_t, out_r]
                    if bsz > target_bsz * 2:
                        outs.append(self.attn_op(
                            q[target_bsz * 2:],
                            k[target_bsz * 2:],
                            v[target_bsz * 2:],
                            transformer_options=transformer_options,
                        ))

                    out = torch.cat(outs, dim=0)
                    return self.output_dropout(self.output_proj(out))
                except Exception as exc:
                    if hasattr(stats, "adapter_attn_failures"):
                        stats.adapter_attn_failures += 1
                    print(f"{prefix} ⚠ adapter self-attn patch failed in {module_name}: {exc}")
                    if _coerce_bool(getattr(stats, "verbose", False)) or _coerce_bool(getattr(stats, "rf_verbose", False)):
                        traceback.print_exc()
                    return orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            return patched_forward

        module.forward = types.MethodType(make_forward(original_forward, name, block_idx_for_module), module)
        setattr(module, "_untwist_adapter_active", True)
        installed += 1

    return matched, installed, restored, patched_names


def uses_reference_branch_kv() -> bool:
    return False
