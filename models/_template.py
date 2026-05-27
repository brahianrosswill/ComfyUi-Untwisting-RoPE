"""
Untwisting RoPE - New Model Adapter Template
============================================

To add a new model:
1. Copy this file and rename it (e.g., `my_model.py`).
2. Update the ARCHITECTURE, DISPLAY_NAME, and SUPPORTED_MODEL_CONFIG_CLASSES.
3. Implement the QKV extraction and concatenation logic inside `patch_attention_modules`.
4. Import your new file in `__init__.py` and add it to `model_adapters`.

RULES:
- NEVER put `print()` statements in this file. Return `patched_names` and let `__init__.py` print them.
- Always return `(matched, installed, restored, patched_names)` from `patch_attention_modules`.
- Pay close attention to sequence slicing (e.g., separating text from image tokens) before applying AdaIN or RoPE.
"""

from __future__ import annotations

import types
import torch
from typing import Any, Dict, List, Optional, Tuple

# Optional: Import specific ComfyUI attention math if your model needs it
# from comfy.ldm.modules.attention import optimized_attention_masked
# from comfy.ldm.flux.math import apply_rope


# ═════════════════════════════════════════════════════════════════════════════
# 1. METADATA & IDENTITY
# ═════════════════════════════════════════════════════════════════════════════

ARCHITECTURE = "my_model"
DISPLAY_NAME = "My New Model"
CONFIG_KEY = "untwisting_rope"

# Which ComfyUI BaseModel classes this adapter applies to
SUPPORTED_MODEL_CONFIG_CLASSES = {"MyModelConfigName"}

# Standard paths to find the actual diffusion UNet/Transformer inside ComfyUI's wrapper
DIFFUSION_ATTR_PATHS = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)

def matches_model(model_info: Dict[str, Any]) -> bool:
    """Tells __init__.py if this adapter should be used for the current model."""
    return str(model_info.get("model_config_class", "")) in SUPPORTED_MODEL_CONFIG_CLASSES

def _get_attr_path(root: Any, attr_path: str) -> Tuple[Any, bool]:
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
    """Locates the raw PyTorch diffusion module inside the ComfyUI model object."""
    for path in DIFFUSION_ATTR_PATHS:
        obj, ok = _get_attr_path(model_patcher, path)
        if ok and obj is not None:
            return obj
    raise RuntimeError(f"Could not find ComfyUI BaseModel.diffusion_model for {DISPLAY_NAME}.")


# ═════════════════════════════════════════════════════════════════════════════
# 2. RUNTIME CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

def default_runtime_cfg(dm: Any | None = None) -> Dict[str, Any]:
    """
    Returns architecture-specific defaults injected into the configuration dictionary.
    Useful if you need to extract fixed dimensions (like head_dim or axes_dims) 
    from the diffusion model `dm` before the patch runs.
    """
    cfg: Dict[str, Any] = {"architecture": ARCHITECTURE}
    
    # Example: If your model has a fixed head_dim or 3D RoPE axes, define them here:
    # cfg["head_dim"] = 128
    # cfg["axes_dims"] = [16, 56, 56]
    
    return cfg

def prepare_reference_conditioning(
    ref_conditioning: Any, 
    dm: Any, 
    device: Any, 
    dtype: Any, 
    stats: Any, 
    label: str = "", 
    helpers: Dict[str, Any] | None = None
) -> Tuple[Any, str]:
    """
    Pre-processes the reference conditioning tensor if necessary (e.g., T5 tokenization).
    For most standard architectures, this is a no-op.
    """
    return ref_conditioning, "not-applicable"

def uses_reference_branch_kv() -> bool:
    """
    Returns True if the architecture natively passes the reference image as K/V 
    (like some IP-Adapters). Standard DiTs usually return False.
    """
    return False


# ═════════════════════════════════════════════════════════════════════════════
# 3. CORE ATTENTION PATCH
# ═════════════════════════════════════════════════════════════════════════════

# Generic AdaIN helper (leave this alone)
def _adain(target: torch.Tensor, style: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    t_mean = target.mean(dim=1, keepdim=True)
    s_mean = style.mean(dim=1, keepdim=True)
    t_std = target.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    s_std = style.float().var(dim=1, keepdim=True, unbiased=False).add(eps).sqrt().to(target.dtype)
    return (target - t_mean) / t_std * s_std + s_mean

def _coerce_strength01(value: Any, default: float = 0.0) -> float:
    try:
        strength = float(value)
    except Exception:
        strength = float(default)
    if not torch.isfinite(torch.tensor(strength)):
        strength = float(default)
    return max(0.0, min(1.0, strength))

def patch_attention_modules(dm: Any, stats: Any, helpers: Dict[str, Any] | None = None) -> Tuple[int, int, int, List[str]]:
    """
    The main hook. Iterates through the model, finds the attention blocks, and overwrites `forward`.
    
    Returns exactly 4 items: (matched, installed, restored, patched_names)
    Do NOT print here. Let __init__.py handle logging.
    """
    helpers = helpers or {}
    
    # Import math helpers dynamically provided by __init__.py
    lerp = helpers.get("lerp")
    cross_batch_adain_qk = helpers.get("cross_batch_adain_qk")
    build_frequency_scale_vector = helpers.get("build_frequency_scale_vector")
    
    matched = installed = restored = 0
    patched_names: List[str] = []

    # 1. Iterate through the model to find target attention layers
    for name, module in dm.named_modules():
        
        # TODO: Define your criteria for what counts as an attention layer.
        # e.g., if "attn" not in name: continue
        if not hasattr(module, "qkv") or not hasattr(module, "forward"):
            continue

        matched += 1
        patched_names.append(name)

        # 2. Store the original forward pass safely
        if hasattr(module, "_untwist_orig_forward"):
            module.forward = module._untwist_orig_forward
            restored += 1
        else:
            module._untwist_orig_forward = module.forward
            
        original_forward = module._untwist_orig_forward

        # 3. Create the patched forward pass
        def make_forward(orig_fn, module_name):
            def patched_forward(self, *args, **kwargs):
                
                # A. Extract options and check if the effect is enabled
                transformer_options = kwargs.get("transformer_options", {})
                cfg = transformer_options.get(CONFIG_KEY, {})
                if not cfg or not cfg.get("enabled"):
                    return orig_fn(self, *args, **kwargs)

                target_bsz = int(cfg.get("cross_batch_target_batch", 0))
                if target_bsz <= 0:
                    return orig_fn(self, *args, **kwargs)

                # B. Extract x (Assuming x is the first argument, adjust based on model architecture)
                x = args[0] 
                bsz, seqlen, _ = x.shape
                if bsz < target_bsz * 2:
                    return orig_fn(self, *args, **kwargs)

                # C. Check block active range
                block_idx = int(transformer_options.get("block_index", -1))
                active_blocks = cfg.get("active_blocks", set())
                if active_blocks and block_idx not in active_blocks:
                    return orig_fn(self, *args, **kwargs)

                # D. SEQUENCE SLICING (Crucial!)
                # If text and image tokens are concatenated, find where the image starts/ends.
                # Do NOT apply AdaIN or RoPE scaling to text tokens!
                ref_ranges = cfg.get("ref_real_ranges", [])
                if ref_ranges:
                    img_s, img_e = ref_ranges[0]
                else:
                    img_s, img_e = 0, seqlen  # Fallback to whole sequence
                    
                img_s = max(0, min(img_s, seqlen))
                img_e = max(img_s, min(img_e, seqlen))

                if img_e <= img_s:
                    return orig_fn(self, *args, **kwargs)

                if hasattr(stats, "attn_calls"):
                    stats.attn_calls += 1

                # ---------------------------------------------------------------------
                # E. CUSTOM QKV LOGIC (Model Specific)
                # ---------------------------------------------------------------------
                # TODO: Implement how YOUR model extracts Q, K, and V. 
                # Example:
                # xq, xk, xv = torch.split(self.qkv(x), [dim_q, dim_k, dim_v], dim=-1)
                # xq = self.q_norm(xq) ... etc.
                
                # --- PSEUDOCODE FOR UNTWISTING ---
                # 1. Pre-RoPE AdaIN (Only on img_s:img_e tokens)
                # a = float(cfg.get("adain_strength", 0.0))
                # if a > 0:
                #     apply AdaIN from K_reference to K_target...
                
                # 2. Apply RoPE (Model specific implementation)
                # xq, xk = apply_rope(xq, xk, freqs_cis)
                
                # 3. Calculate Frequency Scales
                # progress = float(cfg.get("progress", 0.0))
                # high_scale = lerp(cfg["high_scale_start"], cfg["high_scale_end"], progress)
                # low_scale  = lerp(cfg["low_scale_start"],  cfg["low_scale_end"],  progress)
                # scale_vec = build_frequency_scale_vector(...)
                
                # 4. Scale Reference K and Concatenate to Target
                # ref_k = xk[target_bsz:target_bsz*2, img_s:img_e] * scale_vec
                # ref_v = xv[target_bsz:target_bsz*2, img_s:img_e]
                # k_t_full = torch.cat([xk[:target_bsz], ref_k], dim=1)  # (dim depends on layout)
                # v_t_full = torch.cat([xv[:target_bsz], ref_v], dim=1)
                
                # 5. Calculate Attention
                # out_t = optimized_attention_masked(xq_t, k_t_full, v_t_full...)
                # out_r = optimized_attention_masked(xq_r, xk_r, xv_r...)
                # (Also handle uncond batch if present: xq_e, xk_e, xv_e)
                
                # 6. Post Attention AdaIN (Optional)
                # post_a = _coerce_strength01(cfg.get("post_attention_adain_strength", 0.0))
                # if post_a > 0:
                #     out_t_adain = _adain(out_t, out_r)
                #     out_t = ... blend ...
                
                # 7. Recombine and return
                # final_out = torch.cat([out_t, out_r, out_e], dim=0)
                # return self.out(final_out)
                
                # (Remove this fallback once implemented)
                return orig_fn(self, *args, **kwargs)

            return patched_forward

        # Attach the patch to the module
        module.forward = types.MethodType(make_forward(original_forward, name), module)
        installed += 1

    # Return exactly 4 items so __init__.py can print the results centrally.
    return matched, installed, restored, patched_names
