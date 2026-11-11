# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable GR00T N1.7 loading from the trained checkpoint, without base-model weights."""

import json
from pathlib import Path
from typing import Any

from ._logging import get_logger

logger = get_logger()


def load_groot_model(
    policy_path: str | Path,
    runtime_root: str | Path,
    device: str = "cuda",
    weights_dtype: str = "float32",
) -> tuple[Any, Any, Any, Any]:
    """Load policy, saved processors, and torch without fetching base weights."""
    import torch
    from accelerate import init_empty_weights
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.groot_n1_7 import (
        GR00TN17,
        GR00TN17Config,
        _tie_unused_qwen_lm_head,
    )
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from safetensors import safe_open
    from safetensors.torch import load_file

    root = Path(policy_path).expanduser().resolve()
    assets = Path(runtime_root).expanduser().resolve() / "assets"
    raw = json.loads((root / "config.json").read_text())
    image_keys = sorted(
        k for k, f in raw["input_features"].items() if f["type"] == "VISUAL"
    )
    expected_images = [f"observation.images.{x}" for x in ("front", "left", "wrist")]
    if (
        raw["type"] != "groot"
        or raw["output_features"]["action"]["shape"] != [7]
        or raw["input_features"]["observation.state"]["shape"] != [7]
        or image_keys != expected_images
        or raw.get("use_relative_actions")
    ):
        raise ValueError(
            "Requires the three-camera, 7D absolute-joint GR00T checkpoint"
        )
    config = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    if not isinstance(config, GrootConfig):
        raise ValueError("Checkpoint did not resolve to GrootConfig")
    saved_pre = json.loads((root / "policy_preprocessor.json").read_text())
    saved_pack = next(
        s["config"]
        for s in saved_pre["steps"]
        if s["registry_name"] == "groot_n1_7_pack_inputs_v1"
    )
    # Training and inference both use mapping.get(tag, 0). A saved mapping can
    # deliberately omit new_embodiment; do not replace its trained ID with 10.
    expected_embodiment_id = saved_pack["embodiment_mapping"].get(
        config.embodiment_tag, 0
    )
    if not isinstance(expected_embodiment_id, int) or isinstance(
        expected_embodiment_id, bool
    ):
        raise ValueError("Invalid saved embodiment ID")
    config.device = device
    config.base_model_path = str(assets / "base_model")
    config.model_params_fp32 = weights_dtype == "float32"
    config.use_bf16 = True
    architecture = json.loads((assets / "base_model/config.json").read_text())

    class DeploymentPolicy(GrootPolicy):
        def _create_groot_model(self):
            arch = GR00TN17Config(**architecture)
            arch.use_flash_attention = False
            return GR00TN17(arch, load_backbone_weights=False)

    # Build only the architecture, then stream trained tensors onto the selected device.
    # This avoids simultaneously keeping random weights and the 12.6-GB state dict in RAM.
    with init_empty_weights():
        policy = DeploymentPolicy(config)
    # Tying outside init_empty_weights preserves parameter identity for Accelerate.
    _tie_unused_qwen_lm_head(policy._groot_model.backbone.model)
    weights = root / "model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as stream:
        checkpoint_keys = set(stream.keys())
    expected_keys = set(policy.state_dict())
    groups = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
    tied_aliases = {
        name
        for names in groups.values()
        if set(names) & checkpoint_keys
        for name in names
    }
    missing = expected_keys - checkpoint_keys - tied_aliases
    unexpected = checkpoint_keys - expected_keys
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint architecture mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[weights_dtype]
    # LeRobot's save_model archive has tied-weight metadata rather than HF's format=pt
    # metadata. Read it with safetensors directly and assign into the empty skeleton.
    state = load_file(str(weights), device=device)
    for names in groups.values():
        present = next((name for name in names if name in state), None)
        if present is not None:
            for name in names:
                state.setdefault(name, state[present])
    policy.load_state_dict(state, strict=True, assign=True)
    del state
    _tie_unused_qwen_lm_head(policy._groot_model.backbone.model)
    policy.to(device=device, dtype=dtype)
    if any(p.is_meta for p in policy.parameters()):
        raise RuntimeError("Checkpoint loading left uninitialized parameters")
    policy.requires_grad_(False).eval()
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=str(root),
        preprocessor_overrides={
            "groot_n1_7_vlm_encode_v1": {
                "model_name": str(assets / "Cosmos-Reason2-2B"),
                "device": device,
            },
            "device_processor": {"device": device},
        },
    )
    pack = next(s for s in pre.steps if type(s).__name__ == "GrootN17PackInputsStep")
    decode = next(
        s for s in post.steps if type(s).__name__ == "GrootActionUnpackUnnormalizeStep"
    )
    effective_embodiment_id = pack.embodiment_mapping.get(config.embodiment_tag, 0)
    if pack.training or effective_embodiment_id != expected_embodiment_id:
        raise ValueError("Wrong inference mode or embodiment ID")
    policy.deployment_embodiment_id = effective_embodiment_id
    if decode.env_action_dim != 7 or decode.libero_gripper_action:
        raise ValueError("Wrong absolute-joint postprocessor")
    for step in (pack, decode):
        if torch.as_tensor(step.stats["action"]["min"]).shape != (7,):
            raise ValueError(
                "Action normalization statistics must have seven dimensions"
            )
    logger.info("%s", f"[model] GR00T N1.7: {root}")
    logger.info(
        "%s",
        f"[model] chunk={config.chunk_size}, state/action=7, views=front,left,wrist, "
        f"embodiment={effective_embodiment_id}",
    )
    logger.info(
        "%s",
        f"[model] weights={dtype}, BF16 forward; saved min/max postprocessor -> absolute SDK units",
    )
    return policy, pre, post, torch
