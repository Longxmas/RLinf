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
    (vvla-owned forward over lerobot-loaded weights) and produces
    ``forward_inputs`` in exactly the openpi convention (``chains``,
    ``denoise_inds``, tokenized prompts) so the actor consumes them unchanged.

Sampling (SDE loop owned by the vvla engine, this class is format glue):
  * the denoise loop is ``vvla.rollout.logprob.flow_sample_with_logprob``; the
    adapter passes a per-timestep ``sigma(t)`` callable that reproduces openpi's
    flow-SDE noise schedule ``sigma(t) = noise_level * sqrt(t / (1 - t))``
    (with openpi's t==1 endpoint clamp, see :func:`_openpi_flow_sde_sigma`) at
    the single randomly selected denoise step, and sigma == 0 (a deterministic
    ODE step in vvla's sampler) elsewhere — matching openpi's ODE-SDE mixed
    sampling where only ``denoise_inds`` is stochastic;
  * the transition mean is supplied via vvla's ``mean_fn`` hook as openpi's
    score-corrected flow-SDE drift (:func:`_openpi_flow_sde_mean`, verified
    algebraically identical to openpi's x0/x1-weight form)::

        mean = x + v*dt - (sigma(t)^2*|dt|/(2t)) * (x + v*(1-t))
        std  = sigma(t)*sqrt(|dt|)

    so the realized behavior kernel equals openpi's exactly; ``prev_logprobs``
    and the actor's ``default_forward`` recompute evaluate the same mean on the
    same ``(x_j, x_{j+1})``, hence the PPO ratio at theta == theta_behavior is
    exactly 1.

Weight sync (actor -> rollout):
  * actor (openpi) keys are prefix-less (``paligemma_with_expert...``,
    ``action_in_proj...``, ``time_mlp_*...``) plus ``value_head...``; this
    class's registered tree is the same graph under ``model.`` plus
    ``value_head``. ``state_dict()`` exports actor-convention keys and
    ``load_state_dict()`` maps them back;
  * ``PatchWeightSyncer`` applies updates by in-place ``copy_`` into the
    tensors returned by ``state_dict()`` and requires the sender/receiver key
    *sets* to match exactly — hence ``state_dict()`` must return storage-
    sharing tensors (plain key remap of ``super().state_dict()``), and
    ``vvla_policy`` must NOT be registered as a submodule (it shares every
    weight module with ``self.model``; registering it would duplicate the whole
    tree under ``vvla_policy._lerobot.model.*`` and break the key-set check).

Input pipeline: the openpi transform pipeline (repack -> LiberoInputs ->
Normalize -> tokenize/resize) is reused verbatim so tokenized prompts and
normalized states are bit-identical to the actor's. The transformed images are
fed straight into a vvla ``Pi05Batch`` — deliberately BYPASSING vvla's
``Pi05Policy.collate``: openpi transforms already resize to 224x224 and the
uint8->[-1,1] conversion below mirrors openpi ``Observation.from_dict``, while
vvla's collate would run lerobot ``_preprocess_images`` (resize + ``*2-1``) a
second time and corrupt the value range.

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

import math
import random
from collections import OrderedDict

import torch
from safetensors import safe_open
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor

# Sigma for the ODE (non-selected) denoise steps: vvla's sampler treats
# sigma(t) == 0 as a deterministic ODE step (no noise, zero logprob term),
# exactly openpi's flow_ode steps in the mixed ODE-SDE sampler.
_ODE_SIGMA = 0.0


def _openpi_flow_sde_mean(
    x: torch.Tensor, v: torch.Tensor, t_val: float, dt: float, sig: float
) -> torch.Tensor:
    """openpi flow_sde transition mean, as a vvla sampler ``mean_fn``.

    Algebraically identical to openpi's ``sample_mean_var_val`` x0/x1-weight
    form: ``x + v*dt - (sigma(t)^2*|dt|/(2t)) * (x + v*(1-t))``. The correction
    term carries a ``sigma^2`` factor, so on sigma == 0 (ODE) steps this
    reduces to the plain Euler mean — one hook serves the whole mixed loop.
    """
    mean = x + v * dt
    if sig > 0.0:
        mean = mean - (sig * sig * abs(dt) / (2.0 * t_val)) * (x + v * (1.0 - t_val))
    return mean

_PI05_KEY_LAYOUT_NOTE = (
    "actor(openpi) checkpoints are prefix-less; this class's tree is the same "
    "graph under 'model.' plus 'value_head'"
)


def _openpi_flow_sde_sigma(t_val: float, num_steps: int, noise_level: float) -> float:
    """openpi's flow-SDE sigma schedule, translated to a scalar function of t.

    Reference (``sample_mean_var_val``, flow_sde branch)::

        timesteps  = linspace(1, 1/N, N) ++ [0]        # t_k = 1 - k/N, k=0..N-1
        denom      = where(timesteps == 1, timesteps[1], timesteps)
        sigma(t_k) = noise_level * sqrt(t_k / (1 - denom_k))

    i.e. sigma(t) = noise_level*sqrt(t/(1-t)) with the t == 1 grid endpoint's
    denominator clamped using t_1 = 1 - 1/N, giving sigma(1) = noise_level*sqrt(N).
    The grid coincides exactly with vvla's ``Pi05Policy.flow_schedule`` (t from
    1 -> 0, dt = -1/N).
    """
    denom = 1.0 - t_val
    if denom <= 0.0:
        denom = 1.0 / num_steps  # openpi's t==1 clamp: 1 - t_1 = 1/N
    return noise_level * math.sqrt(t_val / denom)


# openpi's PaliGemmaWithExpertModel.to_bfloat16_for_selected_params keep-fp32
# list, verbatim: everything else in the backbone (both Gemma towers, the
# SigLIP encoder, the projector, embed_tokens/lm_head) is cast to bf16; the
# vision patch/position embeddings and every text-tower RMSNorm stay fp32.
# Substring matching over parameter names works identically on the lerobot
# tree (same submodule names under paligemma_with_expert).
_OPENPI_KEEP_FP32_SELECTORS = (
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)


def _cast_backbone_bf16_openpi(pwe: nn.Module) -> None:
    """Replicate openpi's selective bf16 cast on the lerobot backbone tree.

    Mirrors ``to_bfloat16_for_selected_params("bfloat16")``, which the native
    openpi rollout applies to ``paligemma_with_expert`` only — the flow-side
    modules outside it (action_in/out_proj, time_mlp_*, state_proj) and the
    value head stay fp32 on both sides, so weight-sync dtypes match key-by-key.
    """
    pwe.to(dtype=torch.bfloat16)
    for name, param in pwe.named_parameters():
        if any(sel in name for sel in _OPENPI_KEEP_FP32_SELECTORS):
            param.data = param.data.to(dtype=torch.float32)


def _install_embed_image_shim(lerobot_policy) -> None:
    """Make lerobot's PaliGemmaWithExpert.embed_image work under either a stock
    or an openpi-patched transformers install (see module docstring)."""
    pwe = lerobot_policy.model.paligemma_with_expert
    pg_model = pwe.paligemma.model
    hidden_scale = pwe.paligemma.config.text_config.hidden_size**0.5
    tok_embed = pg_model.language_model.embed_tokens

    def embed_image(image: torch.Tensor):
        # Feed the vision tower fp32 pixels (openpi convention: images stay
        # fp32 into the fp32 patch embedding; the patched SigLIP encoder
        # handles its own internal precision), and hand the features back in
        # the language-embedding dtype so lerobot's embed_prefix concatenates
        # them with the (possibly bf16) token embeddings — the same effective
        # dtype the native openpi embed_prefix produces. fp32 tree: no-ops.
        out_dtype = tok_embed.weight.dtype
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
    # openpi checkpoints tie embed_tokens to lm_head and ship only the latter;
    # lerobot's embed_tokens is an independent parameter, so materialize the
    # tied weight explicitly (otherwise it silently stays at random init).
    _LM_HEAD = "model.paligemma_with_expert.paligemma.lm_head.weight"
    _EMBED = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    if _EMBED not in state and _LM_HEAD in state:
        state[_EMBED] = state[_LM_HEAD]
    missing, unexpected = lerobot_policy.load_state_dict(state, strict=False)
    # value_head etc. live on the actor, not in the SFT release; tolerate
    # missing aux keys but never silently skip checkpoint tensors.
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys (layout drift?): {unexpected[:5]}")


class VvlaForRLActionPrediction(nn.Module, BasePolicy):
    """Rollout-only pi0.5 policy served by the vvla engine.

    ``predict_action_batch`` mirrors ``OpenPi0ForRLActionPrediction``'s output
    contract field-for-field (``prev_logprobs`` ``[B, action_chunk,
    action_env_dim]``, ``prev_values`` ``[B, 1]``, ``forward_inputs`` with
    ``chains [B, N+1, H, A]`` / ``denoise_inds [B, N]`` / tokenized prompts /
    ``action`` / ``model_action`` / cloned observations); the actor's openpi
    ``default_forward`` consumes them unchanged. Supported configuration:
    ``noise_method == 'flow_sde'`` with ``value_after_vlm`` VLM-pooled value
    (the openpi PPO pi0.5 recipe); joint_logprob / double_layer / NFT / DSRL /
    RLT / suffix-token value are rejected at construction.
    """

    def __init__(self, cfg):
        super().__init__()
        vv = cfg.get("vvla", {}) or {}

        # ---- openpi-side config + transform pipeline (actor-identical) -------
        # Reuses the exact construction in rlinf.models.embodiment.openpi.get_model
        # (same TrainConfig, same norm-stats source, same transform order) minus
        # the PI0Pytorch instantiation.
        import openpi.shared.download as download
        import openpi.transforms as _transforms
        from openpi.training import checkpoints as _checkpoints

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
        from rlinf.models.embodiment.openpi.openpi_action_model import (
            OpenPi0Config,
            OpenPi0ForRLActionPrediction,
        )

        # Unbound-method reuse: obs/transform/value/noise helpers are pure
        # functions of (self.config, self._input_transform, self._output_transform,
        # self.value_head, self.global_step), all of which this class provides.
        self._openpi_ref = OpenPi0ForRLActionPrediction

        opi = cfg.get("openpi", {}) or {}
        config_name = opi.get("config_name", "pi05_libero")
        data_kwargs = cfg.get("openpi_data", None)
        train_config = get_openpi_config(
            config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
        )
        model_config = OpenPi0Config(**train_config.model.__dict__)
        for key, val in opi.items():
            model_config.__dict__[key] = val
        # vvla-scoped noise overrides take precedence over the openpi block.
        for key in ("noise_method", "noise_level", "noise_anneal", "noise_params"):
            if key in vv:
                model_config.__dict__[key] = vv[key]
        # RL geometry comes from the top-level model cfg (single source of truth
        # with the env/worker), matching the openpi yaml interpolations.
        model_config.__dict__["action_chunk"] = int(
            cfg.get("num_action_chunks", model_config.action_chunk)
        )
        model_config.__dict__["action_env_dim"] = int(
            cfg.get("action_dim", model_config.action_env_dim)
        )
        model_config.__dict__["num_steps"] = int(
            cfg.get("num_steps", model_config.num_steps)
        )
        model_config.__dict__["add_value_head"] = bool(
            opi.get("add_value_head", cfg.get("add_value_head", False))
        )
        self.config = model_config

        # ---- supported-configuration guards ----------------------------------
        if not getattr(self.config, "pi05", False):
            raise ValueError("vvla rollout backend supports pi0.5 configs only")
        if self.config.noise_method != "flow_sde":
            raise ValueError(
                f"vvla rollout backend supports noise_method='flow_sde' only, "
                f"got '{self.config.noise_method}'"
            )
        for flag in ("joint_logprob", "double_layer", "use_dsrl", "use_rlt", "is_nft"):
            if getattr(self.config, flag, False):
                raise ValueError(f"vvla rollout backend does not support {flag}=True")
        if self.config.add_value_head and not self.config.value_after_vlm:
            raise ValueError(
                "vvla rollout backend supports the VLM-pooled value head only "
                "(set openpi.value_after_vlm=True); the suffix-token value path "
                "needs openpi's per-step suffix_out"
            )

        # ---- transform pipeline (same lists as openpi get_model) -------------
        checkpoint_dir = download.maybe_download(str(cfg.model_path))
        data_config = train_config.data.create(train_config.assets_dirs, model_config)
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
        repack_transforms = _transforms.Group()
        self.setup_wrappers(
            transforms=[
                *repack_transforms.inputs,
                _transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.data_transforms.outputs,
                *repack_transforms.outputs,
            ],
        )

        # ---- lerobot-loaded weights, vvla-owned forward -----------------------
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        base_cfg = PreTrainedConfig.from_pretrained(vv.get("base_config_path"))
        lerobot_policy = PI05Policy(base_cfg)
        _load_sft_weights(lerobot_policy, cfg.model_path)
        # Match the native rollout's precision regime: openpi's builder applies
        # to_bfloat16_for_selected_params("bfloat16") to the backbone after
        # loading. vvla.precision: "bfloat16" (default, actor-identical) or
        # "float32" (full-fp32 fallback for precision-regression comparisons).
        precision = vv.get("precision", "bfloat16")
        if precision == "bfloat16":
            _cast_backbone_bf16_openpi(lerobot_policy.model.paligemma_with_expert)
        elif precision != "float32":
            raise ValueError(f"vvla.precision must be 'bfloat16' or 'float32', got '{precision}'")
        _install_embed_image_shim(lerobot_policy)
        # registered as a submodule so weight sync sees real parameters
        self.model = lerobot_policy.model

        from vvla.policies.factory import make_policy

        # NOT registered as a submodule (object.__setattr__ bypasses nn.Module
        # registration): the vvla policy wraps the very same weight modules as
        # self.model, and registering it would duplicate every parameter under
        # 'vvla_policy._lerobot.model.*' in state_dict — breaking the patch
        # syncer's strict key-set equality with the actor. Device/dtype moves on
        # self.model propagate automatically (shared Parameter objects).
        object.__setattr__(
            self,
            "vvla_policy",
            make_policy(
                "pi05",
                checkpoint=lerobot_policy,
                attention=vv.get("attention", "eager"),
            ),
        )

        # ---- value head (constructed exactly as the openpi actor's) ----------
        proj_width = 2048 if self.config.value_after_vlm else 1024
        if self.config.add_value_head:
            if self.config.config_name in [
                "pi05_maniskill",
                "pi05_libero",
                "pi05_droid_polaris",
            ]:
                value_head_hidden_sizes = (1024, 512, 256)
            else:
                value_head_hidden_sizes = (512, 256, 128)
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
        self.use_vlm_value = bool(
            self.config.value_after_vlm and self.config.add_value_head
        )

        self.logger = get_logger()
        self.global_step = 0
        self._warned_unexpected_keys: set = set()
        # kept for readability at the worker boundary
        self.num_action_chunks = self.config.action_chunk
        self.action_env_dim = self.config.action_env_dim
        self.num_steps = self.config.num_steps

    # ---- reused openpi helpers (unbound-method delegation) -------------------
    # These operate only on attributes this class provides (config, transforms,
    # value_head, global_step); reusing them keeps the obs/normalization/value
    # conventions byte-identical to the actor's.
    def setup_wrappers(self, transforms=(), output_transforms=()):
        return self._openpi_ref.setup_wrappers(self, transforms, output_transforms)

    def obs_processor(self, env_obs):
        return self._openpi_ref.obs_processor(self, env_obs)

    def input_transform(self, obs, transpose=True):
        return self._openpi_ref.input_transform(self, obs, transpose)

    def output_transform(self, outputs):
        return self._openpi_ref.output_transform(self, outputs)

    def precision_processor(self, processed_obs):
        return self._openpi_ref.precision_processor(self, processed_obs)

    def _select_configured_state(self, states):
        return self._openpi_ref._select_configured_state(self, states)

    def get_value_from_vlm(self, prefix_output):
        return self._openpi_ref.get_value_from_vlm(self, prefix_output)

    def get_logprob_norm(self, sample, mu, sigma):
        return self._openpi_ref.get_logprob_norm(self, sample, mu, sigma)

    def _get_noise_level(self, device, dtype, sample_method=None):
        return self._openpi_ref._get_noise_level(self, device, dtype, sample_method)

    def set_global_step(self, global_step):
        """Called by the rollout worker after each weight sync; drives the
        openpi noise-anneal schedule (noise_params) when noise_anneal=True."""
        self.global_step = global_step

    # ---- batch construction ---------------------------------------------------
    def _build_pi05_batch(self, processed_obs):
        """openpi-transformed obs -> vvla ``Pi05Batch`` (collate bypassed).

        The openpi pipeline emits per-camera HWC uint8 images (LiberoInputs +
        ResizeImages keep uint8; 224x224) plus per-camera bool masks and the
        200-token prompt. openpi's ``Observation.from_dict`` converts uint8 to
        float32 in [-1, 1]; lerobot's ``embed_prefix`` (vvla's encode_prefix)
        consumes CHW float [-1, 1] at the same resolution — so the only glue is
        uint8 -> [-1, 1] and HWC -> CHW. Camera order follows the transform's
        dict insertion order (base, left wrist, right wrist), the same order the
        actor's ``_preprocess_observation`` iterates.
        """
        from vvla.policies.pi05.processor_pi05 import Pi05Batch

        # Images stay fp32 regardless of the backbone precision — openpi's
        # Observation.from_dict produces fp32 [-1, 1] pixels and feeds them to
        # the (fp32) vision patch embedding; the bf16 handoff happens inside
        # the tower / at the embed_image boundary (see the shim).
        image_dtype = torch.float32
        image_dict = processed_obs["image"]
        mask_dict = processed_obs["image_mask"]
        images, img_masks = [], []
        for cam, img in image_dict.items():
            if img.dtype == torch.uint8:
                img = img.to(image_dtype) / 255.0 * 2.0 - 1.0
            elif img.dtype != image_dtype:
                # float inputs are already in [-1, 1] (openpi convention)
                img = img.to(image_dtype)
            if img.shape[-1] == 3:  # HWC -> CHW
                img = img.permute(0, 3, 1, 2)
            images.append(img.contiguous())
            img_masks.append(mask_dict[cam].reshape(img.shape[0]).bool())
        tokens = processed_obs["tokenized_prompt"].long()
        masks = processed_obs["tokenized_prompt_mask"].bool()
        request_ids = [str(i) for i in range(tokens.shape[0])]
        return Pi05Batch(images, img_masks, tokens, masks, request_ids)

    # ---- flow-SDE sampling via the vvla engine --------------------------------
    def _make_sigma_schedule(self, sde_step: int | None, num_steps: int, noise_level: float):
        """sigma(t) callable for vvla's sampler: openpi's flow-SDE sigma at the
        single selected denoise step, floor sigma (numerically an ODE step)
        everywhere else — openpi's ODE-SDE mixed sampling."""
        t_grid = [t for t, _ in self.vvla_policy.flow_schedule(num_steps)]
        t_sde = t_grid[sde_step] if sde_step is not None else None
        tol = 0.25 / num_steps  # grid spacing is 1/N; nearest-point match

        def sigma(t_val: float) -> float:
            if t_sde is None or abs(t_val - t_sde) > tol:
                return _ODE_SIGMA
            return _openpi_flow_sde_sigma(t_val, num_steps, noise_level)

        return sigma

    def _openpi_step_logprob(self, chains, v, sde_step, num_steps, noise_level):
        """Elementwise transition logprob of the selected step, in the exact
        openpi flow_sde convention (the actor's ``sample_mean_var_val`` +
        ``get_logprob_norm`` on the same ``(x_j, x_{j+1})``), so the PPO ratio
        at theta == theta_behavior is exactly 1.

        ``v`` is the sampler-recorded velocity ``v(x_j, t_j)`` at the selected
        step (``flow_sample_with_logprob(..., return_velocities=True)``) — the
        very tensor the realized transition used, so no extra ``denoise_step``
        forward is needed.
        """
        t_val, dt = self.vvla_policy.flow_schedule(num_steps)[sde_step]
        delta = abs(dt)
        x_j = chains[:, sde_step]
        x_next = chains[:, sde_step + 1]

        sigma_j = _openpi_flow_sde_sigma(t_val, num_steps, noise_level)
        std = sigma_j * math.sqrt(delta)
        # openpi flow_sde mean == x + v*dt - (sigma^2*delta/(2t)) * (x + v*(1-t))
        # expressed through openpi's own (x0_pred, x1_pred) weights:
        x0_pred = x_j - v * t_val
        x1_pred = x_j + v * (1.0 - t_val)
        x0_weight = 1.0 - (t_val - delta)
        x1_weight = (t_val - delta) - (sigma_j**2) * delta / (2.0 * t_val)
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        x_t_std = torch.full_like(x_t_mean, std)

        log_probs = self.get_logprob_norm(x_next, x_t_mean, x_t_std)
        return log_probs[
            :, : self.config.action_chunk, : self.config.action_env_dim
        ].to(torch.float32)

    @torch.no_grad()
    def _sample_actions_vvla(self, batch, mode="train", compute_values=True):
        """Prefix once + vvla SDE loop; returns the openpi ``sample_actions``
        result contract: actions / chains / prev_logprobs / prev_values /
        denoise_inds."""
        from vvla.rollout.logprob import flow_sample_with_logprob

        device = next(self.parameters()).device
        # x_t / chains / Euler integration run in fp32 (openpi convention:
        # only the tower interiors run bf16; noise, time and the flow state
        # stay fp32 under either backbone precision).
        dtype = torch.float32
        bsize = batch.batch_size
        num_steps = self.config.num_steps

        prefix = self.vvla_policy.encode_prefix(batch, return_hidden=self.use_vlm_value)

        # Initial noise, same shape/distribution as openpi's sample_noise
        # (model horizon/dim, not the env chunk); noise dtype follows x.
        x0 = torch.randn(
            bsize,
            self.config.action_horizon,
            self.config.action_dim,
            device=device,
            dtype=dtype,
        )

        # Denoise-step selection, openpi convention: train picks ONE step index
        # shared across the batch (only that transition is stochastic and
        # scored); eval marks -1 (pure ODE, zero logprobs).
        if mode == "train":
            high = num_steps - 2 if self.config.ignore_last else num_steps - 1
            sde_step = random.randint(0, high)
            denoise_inds = torch.full((bsize, num_steps), sde_step, dtype=torch.long)
        else:
            sde_step = None
            denoise_inds = torch.full((bsize, num_steps), -1, dtype=torch.long)

        noise_level = float(
            self._get_noise_level(device=torch.device("cpu"), dtype=torch.float32)
        )
        sigma_fn = self._make_sigma_schedule(sde_step, num_steps, noise_level)
        # vvla owns the loop; with the openpi mean hook the realized transitions
        # are exactly openpi's mixed ODE-SDE kernel. The [B, num_steps] per-step
        # logprobs vvla returns are summed over all dims, so they are discarded —
        # prev_logprobs needs the elementwise openpi convention, computed below
        # from the sampler-recorded per-step velocities (no extra forward).
        actions, _vvla_logprob, chains, velocities = flow_sample_with_logprob(
            self.vvla_policy,
            prefix,
            x0,
            num_steps,
            sigma=sigma_fn,
            return_trajectory=True,
            per_step=True,
            mean_fn=_openpi_flow_sde_mean,
            return_velocities=True,
        )

        if sde_step is not None:
            prev_logprobs = self._openpi_step_logprob(
                chains, velocities[:, sde_step], sde_step, num_steps, noise_level
            )
        else:
            # eval: openpi's ODE std is 0 and get_logprob_norm masks it to zeros
            prev_logprobs = torch.zeros(
                bsize,
                self.config.action_chunk,
                self.config.action_env_dim,
                device=device,
                dtype=torch.float32,
            )

        if self.use_vlm_value:
            values = self.get_value_from_vlm(prefix.last_hidden)[:, None]
        else:
            values = torch.zeros(bsize, 1, device=device, dtype=torch.float32)

        return {
            "actions": actions,
            "chains": chains,
            "prev_logprobs": prev_logprobs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    # ---- BasePolicy interface -------------------------------------------------
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def predict_action_batch(
        self,
        env_obs,
        mode: str = "train",
        compute_values=True,
        **kwargs,
    ):
        """openpi-contract rollout step: env obs -> (env actions, RL extras)."""
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        batch = self._build_pi05_batch(processed_obs)

        outputs = self._sample_actions_vvla(
            batch, mode=mode, compute_values=compute_values
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": processed_obs["state"]}
        )["actions"]

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            # "action" is the env-executed action; "model_action" the raw model
            # output (pre output_transform) — same convention as openpi.
            "action": actions.reshape(actions.shape[0], -1).contiguous(),
            "model_action": outputs["actions"]
            .reshape(outputs["actions"].shape[0], -1)
            .contiguous(),
        }
        # Clone observations to avoid cross-step reference issues.
        cloned_obs = copy_dict_tensor(
            {k: v for k, v in to_process_obs.items() if k != "prompt"}
        )
        forward_inputs.update(cloned_obs)

        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "vvla is a rollout backend; keep actor.model.model_type=openpi for training"
        )

    # ---- weight sync: openpi(actor) <-> lerobot(this) key mapping -------------
    # Contract with the syncers:
    #   * PatchWeightSyncer: requires receiver state_dict() keys to EXACTLY equal
    #     the actor's get_rollout_state_dict() keys, and applies updates by
    #     in-place copy_ into the returned tensors -> the remap below must not
    #     clone/cast (super().state_dict() returns storage-sharing views).
    #   * BucketWeightSyncer: calls load_state_dict(bucket, strict=False) with
    #     partial actor-key dicts -> missing keys are expected (not logged);
    #     unexpected keys indicate tree drift and are logged once each.
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        inner = super().state_dict(prefix=prefix, keep_vars=keep_vars)
        marker = prefix + "model."
        out = destination if destination is not None else OrderedDict()
        for key, value in inner.items():
            if key.startswith(marker):
                out[prefix + key[len(marker) :]] = value
            else:
                out[key] = value  # value_head.* (and any future aux heads)
        return out

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("model.") or key.startswith("value_head."):
                # already in this class's internal convention (e.g. a checkpoint
                # saved from this class)
                mapped[key] = value
            else:
                mapped["model." + key] = value  # actor(openpi) convention
        result = super().load_state_dict(mapped, strict=strict, assign=assign)
        new_unexpected = [
            k for k in result.unexpected_keys if k not in self._warned_unexpected_keys
        ]
        if new_unexpected:
            self._warned_unexpected_keys.update(new_unexpected)
            self.logger.warning(
                "[vvla weight-sync] dropped unexpected keys (openpi<->lerobot "
                f"tree drift? {_PI05_KEY_LAYOUT_NOTE}): {new_unexpected[:8]}"
                + (" ..." if len(new_unexpected) > 8 else "")
            )
        return result


def get_model(cfg, torch_dtype=None):
    # Match the numerics regime of the native rollout: openpi's PI0Pytorch
    # enables TF32 matmuls globally in its __init__; without this the vvla
    # rollout runs true-fp32 matmuls (~4x slower on tensor-core GPUs), which
    # would skew any speed comparison and differ from the actor's regime.
    torch.set_float32_matmul_precision("high")
    model = VvlaForRLActionPrediction(cfg)
    return model.eval()
