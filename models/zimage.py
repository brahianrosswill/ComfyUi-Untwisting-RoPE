from __future__ import annotations

from typing import Any

ARCHITECTURE = "zimage"
DISPLAY_NAME = "Z-Image"
CONFIG_KEY = "untwisting_rope"

# ComfyUI declares Z-Image Turbo as comfy.supported_models.ZImage.
# Pixel-space Z-Image is a different ComfyUI class and should get its own adapter
# if/when this patch supports its diffusion object explicitly.
COMFY_MODEL_CONFIG_CLASS = "ZImage"
DIFFUSION_ATTR_PATHS = (
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
    "diffusion_model",
)


def matches_model(model_info: dict[str, Any]) -> bool:
    """Select Z-Image only from ComfyUI's explicit MODEL metadata."""
    return str(model_info.get("model_config_class", "")) == COMFY_MODEL_CONFIG_CLASS


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
    raise RuntimeError("Could not find ComfyUI BaseModel.diffusion_model for Z-Image.")


def is_joint_attention(module: Any) -> bool:
    """Return True for the Z-Image joint-attention module shape."""
    return (
        hasattr(module, "qkv") and hasattr(module, "out")
        and hasattr(module, "q_norm") and hasattr(module, "k_norm")
        and hasattr(module, "n_local_heads") and hasattr(module, "n_local_kv_heads")
        and hasattr(module, "head_dim")
        and callable(getattr(module, "forward", None))
    )


def is_main_layers_attention_name(name: str, min_layer: int = 0, max_layer: int = 29) -> bool:
    """Z-Image attention modules are named layers.N.attention."""
    parts = str(name).split(".")
    if len(parts) != 3:
        return False
    if parts[0] != "layers" or parts[2] != "attention":
        return False
    try:
        idx = int(parts[1])
    except Exception:
        return False
    return int(min_layer) <= idx <= int(max_layer)


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    """Architecture-specific cfg fields merged into the main runtime cfg."""
    return {"architecture": ARCHITECTURE}


def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 29) -> bool:
    return is_main_layers_attention_name(name, min_layer, max_layer)


def prepare_reference_conditioning(ref_conditioning: Any, dm: Any, device: Any, dtype: Any, stats: Any, label: str = "", helpers: dict[str, Any] | None = None):
    return ref_conditioning, "not-applicable"


def patch_attention_modules(dm: Any, stats: Any, helpers: dict[str, Any] | None = None):
    helpers = helpers or {}
    if callable(helpers.get("patch_context_refiner_mask_modules")):
        helpers["patch_context_refiner_mask_modules"](dm, stats)
    if callable(helpers.get("patch_patchify_and_embed")):
        helpers["patch_patchify_and_embed"](dm, stats)
    if callable(helpers.get("patch_joint_attention_modules")):
        return helpers["patch_joint_attention_modules"](dm, stats)
    return None


def uses_reference_branch_kv() -> bool:
    return True
