"""Template for a metadata-only UntwistingRoPE model adapter.

Copy this file to ``models/<your_model>.py`` and rename the constants below.
Keep this file private as ``_template.py``: the adapter registry ignores modules
whose filename starts with ``_``.

Design rule
-----------
Adapter selection must use ComfyUI's explicit MODEL metadata only.

Do:
    - Match ``model_info["model_config_class"]`` against ComfyUI
      ``supported_models`` class names.
    - Group variants in one adapter when ComfyUI maps them to the same
      architecture/config class family.
    - Use ``unet_config`` values only after matching, for runtime behavior.

Do not:
    - Guess the architecture from diffusion-model attributes.
    - Guess from attention-module names.
    - Guess from tensor shapes, layer counts, or method presence.
    - Fall back to structural probing when metadata does not match.

Why
---
ComfyUI already resolves the checkpoint to a concrete ``supported_models`` class.
Its loaded ``BaseModel`` stores that object on ``model.model_config`` and the
diffusion module on ``model.diffusion_model``. This adapter should trust that
metadata instead of re-detecting the architecture.
"""

from __future__ import annotations

from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Required adapter identity
# ---------------------------------------------------------------------------

# Internal architecture key written into transformer_options config.
# Use a stable lowercase identifier, usually the filename without ".py".
ARCHITECTURE = "replace_me"

# User-facing label for logs / diagnostics.
DISPLAY_NAME = "Replace Me"

# The exact ComfyUI supported_models class names this adapter supports.
#
# Examples:
#   Flux dev + Flux schnell:
#       SUPPORTED_MODEL_CONFIG_CLASSES = {"Flux", "FluxSchnell"}
#
#   Flux2 dev + Flux2 Klein, if ComfyUI maps both to class Flux2:
#       SUPPORTED_MODEL_CONFIG_CLASSES = {"Flux2"}
#
#   Z-Image latent-space only:
#       SUPPORTED_MODEL_CONFIG_CLASSES = {"ZImage"}
#
# Do not put broad config keys here, such as image_model="lumina2", because
# multiple ComfyUI classes may share those fields while needing different hooks.
SUPPORTED_MODEL_CONFIG_CLASSES: set[str] = {
    "ReplaceWithComfySupportedModelClass",
}


# ---------------------------------------------------------------------------
# Small metadata helpers
# ---------------------------------------------------------------------------

_DIFFUSION_ATTR_PATHS = (
    # Common ComfyUI ModelPatcher -> BaseModel locations.
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",

    # Useful in tests, or if the caller passes the BaseModel directly.
    "diffusion_model",
)


def _get_attr_path(root: Any, attr_path: str) -> tuple[Any, bool]:
    """Safely read a dotted attribute path."""
    obj = root
    for part in attr_path.split("."):
        if obj is None or not hasattr(obj, part):
            return None, False
        try:
            obj = getattr(obj, part)
        except Exception:
            return None, False
    return obj, True


def _class_name(value: Any) -> str:
    return type(value).__name__ if value is not None else ""


def _as_set(values: Iterable[str]) -> set[str]:
    return {str(v) for v in values if str(v)}


# ---------------------------------------------------------------------------
# Required hooks
# ---------------------------------------------------------------------------

def matches_model(model_info: dict[str, Any]) -> bool:
    """Return True only for explicit ComfyUI model_config class matches.

    ``model_info`` is produced by the adapter registry from ComfyUI's loaded
    MODEL object. The important key is:

        model_info["model_config_class"]

    which should be the class name from ``comfy.supported_models`` such as
    "Flux", "FluxSchnell", "Flux2", "ZImage", or "Anima".

    Keep this strict. If it returns False, the adapter should not be used.
    """
    return str(model_info.get("model_config_class", "")) in SUPPORTED_MODEL_CONFIG_CLASSES


def find_diffusion_model(model_patcher: Any) -> Any:
    """Return ComfyUI's already-selected ``BaseModel.diffusion_model``.

    This function is intentionally only a lookup. It must not verify the model
    by checking attributes like ``layers``, ``blocks``, ``patchify_and_embed``,
    q/k/v projections, tensor shapes, or module names.

    Adapter selection already happened through ``matches_model()``.
    """
    for path in _DIFFUSION_ATTR_PATHS:
        obj, ok = _get_attr_path(model_patcher, path)
        if ok and obj is not None:
            return obj

    raise RuntimeError(
        f"Could not find ComfyUI BaseModel.diffusion_model for {DISPLAY_NAME}."
    )


# ---------------------------------------------------------------------------
# Optional runtime config hook
# ---------------------------------------------------------------------------

def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    """Return architecture-specific config merged into transformer_options.

    Use this for values your attention patches need at runtime.

    Good:
        - Static architecture labels.
        - RoPE axis dimensions if they are known for this architecture.
        - Ranges or feature flags required by your own patch code.

    Avoid:
        - New architecture detection.
        - Structural checks that decide whether this adapter should be active.
    """
    return {
        "architecture": ARCHITECTURE,
    }


# ---------------------------------------------------------------------------
# Optional reference-conditioning hook
# ---------------------------------------------------------------------------

def prepare_reference_conditioning(
    ref_conditioning: Any,
    dm: Any,
    device: Any,
    dtype: Any,
    stats: Any = None,
    label: str = "",
    helpers: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Optionally adapt reference CONDITIONING before RF/reference use.

    Return:
        (possibly_modified_conditioning, status_string)

    Implement this only when the model requires architecture-specific text
    preprocessing. For example, an adapter may need to run a ComfyUI diffusion
    method that converts raw text embeddings into the final cross-attention
    shape.

    Keep it deterministic and safe:
        - Do not mutate the input in place unless that is intentional.
        - Use ``torch.inference_mode()`` inside expensive model calls.
        - Return the original conditioning with a clear status string when the
          required metadata is absent.
    """
    return ref_conditioning, "not-applicable"


# ---------------------------------------------------------------------------
# Optional attention-name helpers
# ---------------------------------------------------------------------------

def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """Return True for attention module names this adapter should patch.

    Replace this with the naming pattern used by the ComfyUI diffusion module
    after metadata has selected this adapter.

    Example patterns:
        - "layers.N.attention"
        - "blocks.N.self_attn"
        - "double_blocks.N.img_attn"

    This is not architecture detection; it is patch targeting after the adapter
    has already been selected.
    """
    parts = str(name).split(".")
    if len(parts) != 3:
        return False

    # Example placeholder: layers.N.attention
    if parts[0] != "layers" or parts[2] != "attention":
        return False

    try:
        idx = int(parts[1])
    except Exception:
        return False

    return int(min_layer) <= idx <= int(max_layer)


def is_joint_attention(module: Any) -> bool:
    """Optional module predicate for patch code.

    Use this only after metadata-selected adapter activation, never for adapter
    selection. Keep the check as narrow as the patch implementation needs.
    """
    return False


def uses_reference_branch_kv() -> bool:
    """Whether this architecture expects separate reference-branch K/V handling."""
    return False


# ---------------------------------------------------------------------------
# Optional attention patch hook
# ---------------------------------------------------------------------------

def patch_attention_modules(
    dm: Any,
    stats: Any,
    helpers: dict[str, Any] | None = None,
) -> Any:
    """Patch the model-specific attention modules.

    ``helpers`` is provided by the top-level node and may contain reusable patch
    functions. Prefer helpers over duplicating large generic patch logic.

    Common patterns:
        if callable(helpers.get("patch_joint_attention_modules")):
            return helpers["patch_joint_attention_modules"](dm, stats)

        if callable(helpers.get("patch_patchify_and_embed")):
            helpers["patch_patchify_and_embed"](dm, stats)

    Keep all architecture-specific module names and monkey patches in this file.
    The top-level node should stay model-neutral.
    """
    helpers = helpers or {}

    # Replace with this adapter's actual patch sequence.
    #
    # Example:
    # if callable(helpers.get("patch_joint_attention_modules")):
    #     return helpers["patch_joint_attention_modules"](dm, stats)

    return None


# ---------------------------------------------------------------------------
# Optional diagnostic helper
# ---------------------------------------------------------------------------

def describe_match(model_info: dict[str, Any]) -> str:
    """Return a concise diagnostic string for logs or error messages."""
    model_config_class = str(model_info.get("model_config_class", ""))
    unet_config = model_info.get("unet_config", {})
    image_model = ""
    if isinstance(unet_config, dict):
        image_model = str(unet_config.get("image_model", ""))
    else:
        image_model = str(model_info.get("image_model", ""))

    supported = ", ".join(sorted(SUPPORTED_MODEL_CONFIG_CLASSES))
    return (
        f"{DISPLAY_NAME}: model_config_class={model_config_class!r}, "
        f"image_model={image_model!r}, supported_classes={{{supported}}}"
    )
