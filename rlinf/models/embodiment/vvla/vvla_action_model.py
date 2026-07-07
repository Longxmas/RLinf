# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vvla-backed pi0.5 rollout policy.

Design (rollout-side swap):
  * the FSDP actor keeps the reference ``openpi`` implementation — training,
    ``default_forward`` logprob recompute, and the value head are unchanged;
  * this class serves *rollout generation only*: it wraps a vvla ``Pi05Policy``
    (vvla-owned forward over lerobot-loaded weights) and must produce
    ``forward_inputs`` in exactly the openpi convention (``chains``,
    ``denoise_inds``, tokenized prompts) so the actor consumes them unchanged.

Weight/checkpoint notes (validated on the RLinf-Pi05-LIBERO-SFT release):
  * the checkpoint's ``model.safetensors`` uses the openpi (prefix-less) key
    layout; lerobot's ``PI05Policy`` state dict is the same tree under a
    ``model.`` prefix, so loading reduces to key prefixing;
  * the policy config (shapes/variants) comes from a lerobot-format base
    checkpoint directory (``vvla.base_config_path``, e.g. lerobot/pi05_base) —
    the SFT release ships no lerobot config.json.

Environment note: openpi's ``transformers_replace`` patch makes
``PaliGemmaModel.get_image_features`` return the raw projected feature tensor,
while lerobot 0.5.1 expects a ``BaseModelOutputWithPooling`` whose
``pooler_output`` it re-scales by ``sqrt(text_hidden)`` (stock transformers
pre-divides by the same factor — the round trip is a net no-op).
``_install_embed_image_shim`` bridges the two conventions bit-exactly by
overriding ``embed_image`` on the wrapped instance.
"""

import torch
from safetensors import safe_open
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy


def _install_embed_image_shim(lerobot_policy) -> None:
    """Make lerobot's PaliGemmaWithExpert.embed_image work under either a stock
    or an openpi-patched transformers install (see module docstring)."""
    pwe = lerobot_policy.model.paligemma_with_expert
    pg_model = pwe.paligemma.model
    hidden_scale = pwe.paligemma.config.text_config.hidden_size**0.5

    def embed_image(image: torch.Tensor):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        feats = pg_model.get_image_features(image)
        if not isinstance(feats, torch.Tensor):
            # stock transformers: pooler_output = projector(...) / sqrt(hidden)
            feats = feats.pooler_output * hidden_scale
        # openpi-patched transformers already returns the unscaled projector
        # output, which equals lerobot's (pooler_output * sqrt(hidden)).
        if feats.dtype != out_dtype:
            feats = feats.to(out_dtype)
        return feats

    pwe.embed_image = embed_image


def _load_sft_weights(lerobot_policy, model_path: str) -> None:
    """Load an openpi-layout safetensors checkpoint into a lerobot PI05Policy."""
    import glob
    import os

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {model_path}")
    state = {}
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                state["model." + k] = sf.get_tensor(k)
    missing, unexpected = lerobot_policy.load_state_dict(state, strict=False)
    # value_head etc. live on the actor, not in the SFT release; tolerate
    # missing aux keys but never silently skip checkpoint tensors.
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys (layout drift?): {unexpected[:5]}")


class VvlaForRLActionPrediction(nn.Module, BasePolicy):
    """Rollout-only pi0.5 policy served by the vvla engine.

    Implemented so far: model build (config from a lerobot base checkpoint,
    openpi-layout SFT weight load, patched-transformers shim, vvla wrap).
    Pending (tracked in the fork's integration notes): the openpi-convention
    flow-SDE sampler over vvla's ``encode_prefix``/``denoise_step`` producing
    ``chains``/``denoise_inds``/``prev_logprobs``, the openpi transform
    pipeline reuse for tokenize/normalize, the VLM-mean-token value head, and
    the openpi<->lerobot state-dict name mapping for weight sync.
    """

    def __init__(self, cfg):
        super().__init__()
        vv = cfg.get("vvla", {})

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        base_cfg = PreTrainedConfig.from_pretrained(vv.get("base_config_path"))
        lerobot_policy = PI05Policy(base_cfg)
        _load_sft_weights(lerobot_policy, cfg.model_path)
        _install_embed_image_shim(lerobot_policy)
        # registered as a submodule so weight sync sees real parameters
        self.model = lerobot_policy.model

        from vvla.policies.factory import make_policy

        self.vvla_policy = make_policy(
            "pi05",
            checkpoint=lerobot_policy,
            attention=vv.get("attention", "eager"),
        )
        self.num_action_chunks = cfg.num_action_chunks
        self.action_env_dim = cfg.action_dim
        self.num_steps = cfg.num_steps

    # ---- BasePolicy interface -------------------------------------------
    def predict_action_batch(self, env_obs=None, mode="train", **kwargs):
        raise NotImplementedError(
            "vvla rollout sampler pending: needs the openpi-convention flow-SDE "
            "loop (chains/denoise_inds/logprobs) over vvla encode_prefix/denoise_step"
        )

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "vvla is a rollout backend; keep actor.model.model_type=openpi for training"
        )


def get_model(cfg, torch_dtype=None):
    model = VvlaForRLActionPrediction(cfg)
    return model.eval()
