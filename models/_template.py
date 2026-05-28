"""
_template.py — Architecture Adapter Template
=============================================
Copy this file, rename it (e.g. my_model.py), and fill in every section marked
TODO. Remove all TODO comments and placeholder raises before shipping.

HOW AN ADAPTER FITS IN
-----------------------
The host node (the ComfyUI node or __init__.py) is model-neutral. It:
  1. Calls matches_model() on every loaded model to select the right adapter.
  2. Calls find_diffusion_model() to unwrap the raw diffusion module.
  3. Calls default_runtime_cfg() to get architecture-specific config fields.
  4. Calls prepare_reference_conditioning() to preprocess reference latents/embeds.
  5. Calls patch_attention_modules() to inject the reference-branch logic.

Your adapter file owns ALL architecture-specific knowledge. The host supplies
only generic helpers (math utilities, config keys) through the `helpers` dict.

CONVENTIONS SEEN ACROSS ALL ADAPTERS
--------------------------------------
- Every public function raises RuntimeError (never silently falls back) in strict mode.
- patch_attention_modules() saves and restores the original forward before patching,
  so the patch is idempotent when called twice.
- The patched forward checks cfg["enabled"] first; when False it calls orig unchanged.
- Batch layout is always [target_batch | reference_batch | optional_uncond_batch].
- stats is a live object: increment stats.attn_calls etc. if the attribute exists.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library / torch imports
# ---------------------------------------------------------------------------
import traceback
import types
from typing import Any, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# TODO: import any ComfyUI math helpers your architecture needs.
# Examples from real adapters:
#   from comfy.ldm.flux.math import apply_rope, attention as flux_attention
#   from comfy.ldm.modules.attention import optimized_attention_masked
# ---------------------------------------------------------------------------


# ===========================================================================
# SECTION 1 — IDENTITY
# ===========================================================================

# Short machine-readable tag; stored in cfg["architecture"] at runtime.
# Must be unique across all adapters in the project.
ARCHITECTURE: str = "my_model"  # TODO: replace with your architecture name

# Human-readable label used in log messages and error strings.
DISPLAY_NAME: str = "My Model"  # TODO: replace with your display name

# Name(s) of the ComfyUI supported_models class that identifies this architecture.
# Found in comfy/supported_models.py.  Use a set if multiple variants share one adapter
# (e.g. ZImage uses {"ZImage", "ZImagePixelSpace", "Lumina2"}).
# Use a plain string constant (COMFY_MODEL_CONFIG_CLASS) when there is exactly one.
SUPPORTED_MODEL_CONFIG_CLASSES: set[str] = {"MyModelClass"}  # TODO

# The config key written into transformer_options so patched forwards can find
# their runtime config.  Must match what the host node writes.
CONFIG_KEY: str = "untwisting_rope"  # change only if your host node uses a different key

# ---------------------------------------------------------------------------
# Candidate attribute paths to unwrap the raw diffusion module from whatever
# wrapper ComfyUI has placed around it.  Tried in order; first hit wins.
# This list is identical across all existing adapters — keep it unless ComfyUI
# ever adds a new wrapper layer.
# ---------------------------------------------------------------------------
_DIFFUSION_ATTR_PATHS: tuple[str, ...] = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)


# ===========================================================================
# SECTION 2 — SAFE ATTRIBUTE HELPERS  (copy verbatim, no need to modify)
# ===========================================================================

def _get_attr_path(root: Any, attr_path: str) -> tuple[Any, bool]:
    """Walk a dotted attribute path safely. Returns (value, True) or (None, False)."""
    obj = root
    for part in attr_path.split("."):
        if obj is None or not hasattr(obj, part):
            return None, False
        try:
            obj = getattr(obj, part)
        except Exception:
            return None, False
    return obj, True


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y", "t")
    return bool(value)


def _coerce_strength01(value: Any, default: float = 0.0) -> float:
    """Parse and validate a [0, 1] float strength parameter."""
    try:
        strength = float(value)
    except Exception as exc:
        raise ValueError(f"Invalid strength value {value!r}; expected a finite float in [0, 1].") from exc
    if not torch.isfinite(torch.tensor(strength)):
        raise ValueError(f"Invalid strength value {value!r}; expected a finite float in [0, 1].")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Invalid strength value {strength!r}; expected value in [0, 1].")
    return strength


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _adain(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Instance-normalise target to match style statistics (seq-dim = dim 1)."""
    t_mean = target.mean(dim=1, keepdim=True)
    s_mean = style.mean(dim=1, keepdim=True)
    t_std = target.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    s_std = style.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    return (target - t_mean) / t_std * s_std + s_mean


# ===========================================================================
# SECTION 3 — REQUIRED HOOKS  (called by the host node)
# ===========================================================================

def matches_model(model_info: dict[str, Any]) -> bool:
    """Return True iff this adapter handles the loaded model.

    The host iterates all registered adapters and calls this on each.
    model_info is a dict produced by ComfyUI with at least:
      - "model_config_class": str  (name of the class in supported_models.py)
      - "unet_config": dict        (raw unet config, varies per model)

    Keep the implementation simple: only trust model_config_class.
    """
    return str(model_info.get("model_config_class", "")) in SUPPORTED_MODEL_CONFIG_CLASSES


def is_model_identity(model_info: dict[str, Any]) -> bool:
    """Backward-compatible alias kept for older callers. Do not remove."""
    return matches_model(model_info)


def find_diffusion_model(model_patcher: Any) -> Any:
    """Unwrap and return the raw diffusion module from a ComfyUI ModelPatcher.

    Tries each path in _DIFFUSION_ATTR_PATHS in order.
    Raises RuntimeError (never returns None) — the host relies on this guarantee.
    """
    for path in _DIFFUSION_ATTR_PATHS:
        obj, ok = _get_attr_path(model_patcher, path)
        if ok and obj is not None:
            return obj
    raise RuntimeError(f"Could not find ComfyUI BaseModel.diffusion_model for {DISPLAY_NAME}.")


# ===========================================================================
# SECTION 4 — ARCHITECTURE-SPECIFIC PARAMETER READERS
# ===========================================================================
# These are called by default_runtime_cfg() below.
# Add or remove functions to match what your model exposes.

def _axes_dims_from_dm(dm: Any) -> list[int]:
    """Read the RoPE axis dimensions from the diffusion module.

    TODO: adapt to your model's actual attribute layout.

    FLUX.2 example:
        params = dm.params
        return list(params.axes_dim)

    Anima example (derived from head_dim):
        hd = int(dm.blocks[0].self_attn.head_dim)
        dim_h = (hd // 6) * 2
        return [hd - 2*dim_h, dim_h, dim_h]

    Z-Image example:
        return list(dm.axes_dims)
    """
    raise NotImplementedError(f"{DISPLAY_NAME}: implement _axes_dims_from_dm().")


def _head_dim_from_dm(dm: Any) -> int:
    """Read head_dim from the diffusion module.

    TODO: adapt to your model's actual attribute layout.

    FLUX.2 example:
        return dm.params.hidden_size // dm.params.num_heads

    Anima example:
        return int(dm.blocks[0].self_attn.head_dim)
    """
    raise NotImplementedError(f"{DISPLAY_NAME}: implement _head_dim_from_dm().")


# ===========================================================================
# SECTION 5 — RUNTIME CONFIG
# ===========================================================================

def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    """Return architecture-specific fields merged into the main runtime config.

    The host calls this once per generation run and stores the result under
    transformer_options[CONFIG_KEY].  Always include "architecture".

    If your model needs head_dim or axes_dims at patch time, read them here and
    store them in cfg so the patched forward doesn't have to re-read them.

    For simple architectures that don't need dm at all, set dm=None and ignore it
    (see Z-Image, which only populates cfg["architecture"]).
    """
    cfg: dict[str, Any] = {"architecture": ARCHITECTURE}

    if dm is None:
        # TODO: decide whether your adapter requires dm or not.
        # If it does, raise here. If not, return cfg as-is.
        return cfg

    # TODO: populate architecture-specific fields, for example:
    # cfg["head_dim"]  = _head_dim_from_dm(dm)
    # cfg["axes_dims"] = _axes_dims_from_dm(dm)
    #
    # For models whose attention sees only image tokens (no text prefix),
    # also set the AdaIN image-token range so the generic helper knows where
    # image tokens live in the sequence:
    # cfg["target_qk_adain_ranges"] = [(0, 2 ** 31 - 1)]  # clamped at runtime

    return cfg


# ===========================================================================
# SECTION 6 — MODULE IDENTIFICATION HELPERS
# ===========================================================================
# These let the host (and patch_attention_modules) locate which submodules to patch.

def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """Return True iff `name` is a module path that should be patched.

    TODO: replace the body with your architecture's naming scheme.

    Naming scheme examples:
      Anima   — "blocks.N.self_attn"     → parts = ["blocks", N, "self_attn"]
      FLUX.2  — "double_blocks.N.img_attn" / "single_blocks.N"
      Z-Image — "layers.N.attention" / "noise_refiner.N.attention"
    """
    parts = str(name).split(".")
    # TODO: validate parts length, prefix, suffix, and numeric index.
    # Example for "blocks.N.self_attn":
    #   if len(parts) != 3: return False
    #   if parts[0] != "blocks" or parts[2] != "self_attn": return False
    #   try:
    #       idx = int(parts[1])
    #   except Exception:
    #       return False
    #   return min_layer <= idx <= max_layer
    raise NotImplementedError(f"{DISPLAY_NAME}: implement is_attention_name().")


def is_attention_module(module: Any) -> bool:
    """Return True iff `module` is a patchable attention object.

    Check for the specific attributes your patched forward will actually use.
    This guards against patching the wrong module if ComfyUI renames things.

    TODO: fill in the attribute names your forward accesses on `self`.

    Anima example:
        required = ("q_proj", "k_proj", "v_proj", "attn_op", "n_heads", "head_dim", ...)
        return all(hasattr(module, a) for a in required)
    """
    raise NotImplementedError(f"{DISPLAY_NAME}: implement is_attention_module().")


def block_index_from_name(name: str) -> int:
    """Extract the integer block index from a module name. Used for active_blocks filtering.

    TODO: implement for your naming scheme.

    Anima example: "blocks.3.self_attn" → 3
    FLUX.2 example: "double_blocks.7.img_attn" → 7
    """
    parts = str(name).split(".")
    # TODO: find and return the integer layer index.
    raise NotImplementedError(f"{DISPLAY_NAME}: implement block_index_from_name().")


# ===========================================================================
# SECTION 7 — REFERENCE CONDITIONING PRE-PROCESSING  (optional)
# ===========================================================================

def prepare_reference_conditioning(
    ref_conditioning: Any,
    dm: Any,
    device: Any,
    dtype: Any,
    stats: Any,
    label: str = "",
    helpers: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Pre-process the reference conditioning before the generation loop.

    Most adapters return (ref_conditioning, "not-applicable") unchanged.
    Only override if your architecture requires a forward pass on dm to convert
    raw embeddings before they can be used as reference conditioning
    (see anima.py for a real example that calls dm.preprocess_text_embeds()).

    Returns:
        (processed_conditioning, status_string)
    """
    # TODO: if your model needs preprocessing, implement it here.
    # Otherwise leave this as-is.
    return ref_conditioning, "not-applicable"


# ===========================================================================
# SECTION 8 — ATTENTION MODULE PATCHING  (core of the adapter)
# ===========================================================================

def patch_attention_modules(
    dm: Any,
    stats: Any,
    helpers: dict[str, Any] | None = None,
) -> tuple[int, int, int, list[str]]:
    """Monkey-patch every attention module in dm with the reference-injection forward.

    Parameters
    ----------
    dm      Raw diffusion module returned by find_diffusion_model().
    stats   Live stats object. Increment stats.attn_calls etc. when the attribute exists.
    helpers Dict of callable helpers provided by the host node.  Required helpers differ
            per adapter — declare what you need and raise if they're missing.

    Returns
    -------
    (matched, installed, restored, patched_names)
      matched       — number of modules that matched is_attention_name + is_attention_module
      installed     — number of modules that received a new patched forward
      restored      — number of modules whose previous patch was replaced/restored
      patched_names — list of module name strings that were patched
    """
    helpers = helpers or {}

    # ------------------------------------------------------------------
    # TODO: declare the helpers your patched forward needs and validate them.
    # Common helpers supplied by the host:
    #   "lerp"                       — _lerp(a, b, t) -> float
    #   "cross_batch_adain_qk"       — AdaIN on Q/K across target/ref batches
    #   "build_frequency_scale_vector" — builds per-head RoPE scale vector
    #   "config_key"                 — transformer_options key for this adapter's cfg
    #   "patch_context_refiner_mask_modules" — Z-Image specific
    #   "patch_patchify_and_embed"           — Z-Image specific
    # ------------------------------------------------------------------
    required_helpers: tuple[str, ...] = (
        "lerp",
        "cross_batch_adain_qk",
        "build_frequency_scale_vector",
        # TODO: add any others your forward needs
    )
    missing = [name for name in required_helpers if not callable(helpers.get(name))]
    if missing:
        raise RuntimeError(f"{DISPLAY_NAME} adapter missing required helper(s): {missing}")

    lerp                      = helpers["lerp"]
    cross_batch_adain_qk      = helpers["cross_batch_adain_qk"]
    build_frequency_scale_vector = helpers["build_frequency_scale_vector"]
    config_key                = str(helpers.get("config_key", CONFIG_KEY))

    matched   = 0
    installed = 0
    restored  = 0
    patched_names: list[str] = []

    for name, module in dm.named_modules():
        # ---- filter to patchable targets ----
        if not is_attention_name(name, 0, 999):
            continue
        if not is_attention_module(module):
            continue

        matched += 1
        patched_names.append(name)

        # ---- idempotent save/restore of the original forward ----
        # Always restore first so repeated calls don't chain-wrap.
        _ORIG_FWD_ATTR = "_untwist_orig_forward"  # TODO: use a unique attr name per adapter
        if hasattr(module, _ORIG_FWD_ATTR):
            module.forward = getattr(module, _ORIG_FWD_ATTR)
            restored += 1
        else:
            setattr(module, _ORIG_FWD_ATTR, module.forward)
        original_forward = getattr(module, _ORIG_FWD_ATTR)

        # ----------------------------------------------------------------
        # Build the patched forward.
        # Use a factory (make_forward) to close over `orig` and `module_name`
        # so each module gets its own closure — a plain lambda in a loop
        # would capture the loop variable by reference.
        # ----------------------------------------------------------------
        def make_forward(orig, module_name: str):

            # TODO: adjust the signature to match your module's actual forward signature.
            # Examples:
            #   Anima   — (self, x, context=None, rope_emb=None, transformer_options={})
            #   Z-Image — (self, x, x_mask, freqs_cis, transformer_options={})
            #   FLUX.2 double — (self, img, txt, vec, pe, attn_mask=None, ..., transformer_options={})
            def patched_forward(self, x, transformer_options={}):

                # ---- fast-path: adapter disabled ----
                cfg = (
                    transformer_options.get(config_key)
                    if isinstance(transformer_options, dict) else None
                )
                if not cfg or not cfg.get("enabled"):
                    # TODO: pass the correct arguments to orig() for your architecture.
                    return orig(x, transformer_options=transformer_options)

                # ---- validate batch layout ----
                target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                if target_bsz <= 0:
                    raise RuntimeError(
                        f"{DISPLAY_NAME} Untwisting enabled in {module_name}, "
                        f"but cross_batch_target_batch={target_bsz}."
                    )
                if not torch.is_tensor(x) or x.ndim != 3:
                    raise RuntimeError(
                        f"{DISPLAY_NAME} Untwisting expected x as [B,S,C] tensor in "
                        f"{module_name}; got {type(x).__name__} ndim={getattr(x,'ndim',None)}."
                    )

                bsz, seqlen, _ = x.shape
                if bsz < target_bsz * 2:
                    raise RuntimeError(
                        f"{DISPLAY_NAME} Untwisting expected ≥ target+reference batches in "
                        f"{module_name}; bsz={bsz}, target_bsz={target_bsz}."
                    )

                # ---- active_blocks gate ----
                # block_index comes from transformer_options["block_index"] if the host
                # sets it, otherwise derive it from the module name.
                block_idx   = int(transformer_options.get("block_index", block_index_from_name(module_name)))
                active_blocks = cfg.get("active_blocks", set())
                if active_blocks and block_idx not in active_blocks:
                    return orig(x, transformer_options=transformer_options)

                try:
                    if hasattr(stats, "attn_calls"):
                        stats.attn_calls += 1
                    if hasattr(stats, "adapter_attn_calls"):
                        stats.adapter_attn_calls += 1

                    # --------------------------------------------------------
                    # TODO: implement your QKV extraction.
                    # The exact code depends on your model.  Common patterns:
                    #
                    # Pattern A — dedicated proj layers (Anima style):
                    #   q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
                    #
                    # Pattern B — fused QKV linear (Z-Image / Lumina style):
                    #   xq, xk, xv = torch.split(self.qkv(x), [...], dim=-1)
                    #   xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
                    #   ...
                    #   xq = self.q_norm(xq); xk = self.k_norm(xk)
                    #
                    # Pattern C — FLUX single-block fused linear1:
                    #   qkv, mlp = torch.split(self.linear1(normed_x), [...], dim=-1)
                    #   q, k, v = qkv.view(...).permute(2, 0, 3, 1, 4)
                    # --------------------------------------------------------
                    raise NotImplementedError(f"{DISPLAY_NAME}: implement QKV extraction in patched_forward.")

                    # --------------------------------------------------------
                    # TODO: determine the image-token slice within the sequence.
                    # Architecture-specific:
                    #
                    # Anima: entire sequence is image tokens → (0, seqlen)
                    # Z-Image noise_refiner: entire sequence → (0, seqlen)
                    # Z-Image main layers: read from cfg["ref_real_ranges"][0]
                    # FLUX double-stream: read from transformer_options["img_slice"]
                    # --------------------------------------------------------
                    img_s, img_e = 0, seqlen  # TODO: replace with correct slice

                    # --------------------------------------------------------
                    # Pre-RoPE AdaIN on Q/K (and optionally V) image tokens.
                    # Leave these lines as-is if using [B,S,H,D] layout.
                    # For [B,H,S,D] (FLUX) see flux2.py _flux_adain_qkv_for_image_range.
                    # --------------------------------------------------------
                    a = float(cfg.get("adain_strength", 0.0))
                    apply_adain = cfg.get("apply_adain", False) and a > 0.0
                    if apply_adain:
                        # TODO: replace q, k (and optionally v) with AdaIN-adjusted versions.
                        # Example for [B,S,H,D] layout:
                        #   q, k = cross_batch_adain_qk(q, k, cfg, target_bsz, a)
                        pass

                    # --------------------------------------------------------
                    # TODO: apply RoPE if your architecture uses it.
                    # Example for Z-Image / FLUX:
                    #   q, k = apply_rope(q, k, freqs_cis)
                    # Anima applies RoPE inside compute_qkv already.
                    # --------------------------------------------------------

                    # --------------------------------------------------------
                    # Build the frequency-scale vector for reference K injection.
                    # The scale_vec de-emphasises high-frequency (positional) info
                    # from the reference K so it doesn't over-constrain the layout.
                    # --------------------------------------------------------
                    progress   = float(cfg.get("progress", 0.0))
                    high_scale = lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
                    low_scale  = lerp(cfg["low_scale_start"],  cfg["low_scale_end"],  progress)
                    beta       = float(cfg.get("beta", 2.0))

                    # TODO: replace self.head_dim with the correct attribute for your model.
                    head_dim = int(self.head_dim)
                    scale_vec = build_frequency_scale_vector(
                        head_dim,
                        cfg.get("axes_dims") or getattr(dm, "axes_dims", []),
                        high_scale, low_scale, beta,
                        # TODO: use q.device / q.dtype, or k.device / k.dtype
                        x.device, x.dtype,
                        runtime_cfg=cfg,
                    )
                    # Shape for broadcasting — adjust to match your q/k layout:
                    # [B,S,H,D] → scale_vec.view(1, 1, 1, head_dim)
                    # [B,H,S,D] → scale_vec.view(1, 1, 1, head_dim)
                    scale_vec = scale_vec.view(1, 1, 1, head_dim)  # TODO: check layout

                    # --------------------------------------------------------
                    # TARGET STREAM
                    # Concatenate target K/V with scaled reference K/V so the
                    # target denoising step can "see" the reference image.
                    # --------------------------------------------------------
                    # TODO: adapt slicing to your actual q/k/v tensor shapes.
                    # The example below assumes [B, S, H, D] layout.
                    #
                    # ref_k = k[target_bsz:target_bsz*2, img_s:img_e] * scale_vec
                    # ref_v = v[target_bsz:target_bsz*2, img_s:img_e]
                    # k_t_full = torch.cat([k[:target_bsz], ref_k], dim=1)
                    # v_t_full = torch.cat([v[:target_bsz], ref_v], dim=1)
                    # out_t = <your_attention_op>(q[:target_bsz], k_t_full, v_t_full, ...)
                    raise NotImplementedError(f"{DISPLAY_NAME}: implement target stream attention.")

                    # --------------------------------------------------------
                    # REFERENCE STREAM
                    # Reference attends only to itself — no injection.
                    # --------------------------------------------------------
                    # out_r = <your_attention_op>(q[target_bsz:target_bsz*2],
                    #                             k[target_bsz:target_bsz*2],
                    #                             v[target_bsz:target_bsz*2], ...)
                    raise NotImplementedError(f"{DISPLAY_NAME}: implement reference stream attention.")

                    # --------------------------------------------------------
                    # POST-ATTENTION AdaIN (optional)
                    # --------------------------------------------------------
                    post_a = _coerce_strength01(cfg.get("post_attention_adain_strength", 0.0))
                    if post_a > 0.0:
                        # TODO: blend out_t image tokens toward out_r statistics.
                        # out_t_adain = _adain(out_t[:, img_s:img_e], out_r[:, img_s:img_e])
                        # out_t[:, img_s:img_e] = out_t[:, img_s:img_e] * (1-post_a) + out_t_adain * post_a
                        pass

                    # --------------------------------------------------------
                    # EXTRA BATCHES (uncond / classifier-free guidance negative)
                    # --------------------------------------------------------
                    outs = [out_t, out_r]
                    if bsz > target_bsz * 2:
                        # out_e = <your_attention_op>(q[target_bsz*2:], k[target_bsz*2:], v[target_bsz*2:], ...)
                        # outs.append(out_e)
                        raise NotImplementedError(f"{DISPLAY_NAME}: implement extra (uncond) batch attention.")

                    final_out = torch.cat(outs, dim=0)

                    # --------------------------------------------------------
                    # TODO: apply output projection if your architecture requires it.
                    # Anima:   return self.output_dropout(self.output_proj(final_out))
                    # Z-Image: return self.out(final_out)
                    # FLUX double-block: proj is applied outside this module
                    # --------------------------------------------------------
                    return final_out

                except Exception as exc:
                    if hasattr(stats, "adapter_attn_failures"):
                        stats.adapter_attn_failures += 1
                    raise RuntimeError(
                        f"{DISPLAY_NAME} adapter patch failed in {module_name}; "
                        f"strict mode refuses to call original forward after patch failure: {exc}"
                    ) from exc

            return patched_forward

        module.forward = types.MethodType(make_forward(original_forward, name), module)
        # Optional: mark module so callers can detect active patches.
        setattr(module, "_untwist_adapter_active", True)
        installed += 1

    if installed <= 0:
        raise RuntimeError(
            f"{DISPLAY_NAME} adapter patch failed: no compatible attention modules were installed."
        )

    return matched, installed, restored, patched_names


# ===========================================================================
# SECTION 9 — CAPABILITY FLAGS
# ===========================================================================

def uses_reference_branch_kv() -> bool:
    """Return True if this adapter injects reference K/V into the target stream.

    All existing adapters return False here because reference K/V injection is
    handled inside patch_attention_modules() directly.  Return True only if
    your adapter exposes an additional hook-based KV injection path.
    """
    return False


# ===========================================================================
# SECTION 10 — OPTIONAL DIAGNOSTICS
# ===========================================================================

def describe_match(model_info: dict[str, Any]) -> str:
    """Human-readable string summarising why this adapter matched (for logging).

    Optional — implement only if your host node calls it.
    """
    model_config_class = str(model_info.get("model_config_class", ""))
    supported = ", ".join(sorted(SUPPORTED_MODEL_CONFIG_CLASSES))
    return (
        f"{DISPLAY_NAME}: model_config_class={model_config_class!r}, "
        f"supported_classes={{{supported}}}"
    )


# ===========================================================================
# SECTION 11 — __all__
# ===========================================================================
# Keep in sync with every public symbol you export.

__all__ = [
    "ARCHITECTURE",
    "DISPLAY_NAME",
    "SUPPORTED_MODEL_CONFIG_CLASSES",
    "CONFIG_KEY",
    "matches_model",
    "is_model_identity",
    "find_diffusion_model",
    "default_runtime_cfg",
    "is_attention_name",
    "is_attention_module",
    "block_index_from_name",
    "prepare_reference_conditioning",
    "patch_attention_modules",
    "uses_reference_branch_kv",
    "describe_match",
]
