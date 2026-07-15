# Copyright 2026 The RLinf Authors.
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

"""vvla-backed LingBot-VLA rollout policy (RoboTwin GRPO).

Design (rollout-side swap, mirroring the pi0.5 / GR00T vvla adapters):
  * the FSDP actor keeps the reference ``LingbotvlaActionModel`` — training,
    ``default_forward`` log-prob recompute, and (if enabled) the value head are
    unchanged;
  * this class serves *rollout generation only*: it wraps a vvla
    ``LingBotVLAPolicy`` (stock Qwen2.5-VL VL backbone + vendored Qwen2 action
    expert, vvla-owned MoT denoise loop) and produces the same
    ``(env_actions, {prev_logprobs, prev_values, forward_inputs})`` contract the
    native ``predict_action_batch`` returns, so the actor consumes it unchanged.

This class subclasses ``LingbotvlaActionModel`` so the *observation pipeline* is
reused verbatim — ``obs_processor`` / ``_preprocess_observation`` (state reorder
+ normaliser + Qwen image/text processors) / ``_reorder_rep_state_for_model`` /
``_select_env_action_dims`` / ``output_transform`` / ``get_logprob_norm`` are all
inherited, so the tensors the vvla engine sees are bit-identical to the actor's.
Only ``__init__`` (build vvla instead of the native model) and ``sample_actions``
(vvla forward + the flow-SDE kernel) are overridden; ``predict_action_batch`` is
inherited and drives the overridden ``sample_actions``.

Sampling (SDE kernel owned here, glue to the vvla velocity field):
  LingBot integrates BACKWARD time (t: 1 -> 0, x1 action -> x0 noise; pi0-style,
  like pi0.5), so the per-step mean/std follow ``sample_mean_var_val``'s flow_sde
  branch VERBATIM (x0/x1-weight form)::

      timesteps  = cat([linspace(1, 1/N, N), [0]])
      sigma_i    = noise_level * sqrt(t_i / (1 - where(t_i==1, t_1, t_i)))   [i<N]
      x0_pred    = x_t - v_t * t ;  x1_pred = x_t + v_t * (1 - t)
      x0_weight  = 1 - (t - delta)
      x1_weight  = (t - delta) - sigma_i^2 * delta / (2 * t)    # score correction
      x_t_mean   = x0_pred * x0_weight + x1_pred * x1_weight
      x_t_std    = sqrt(delta) * sigma_i
      x_t        = x_t_mean + randn * x_t_std
      log_prob   = get_logprob_norm(x_t, x_t_mean, x_t_std)      # elementwise Gaussian

  ``v_t`` is the vvla policy's ``denoise_step`` velocity. Because the per-step
  mean/std and the elementwise ``get_logprob_norm`` are the native tensor
  expressions applied to the vvla-sampled chains, the actor's ``default_forward``
  recompute on the same chains differs only by the native-vs-vvla velocity (a
  small residual the flow Gaussian log-prob absorbs) -> the PPO ratio at
  theta == theta_behavior stays within the native noise floor.

  ``joint_logprob`` mode is honoured: False -> noise on one random step, the
  selected step's [B, chunk, env_dim] log-prob is returned (the RoboTwin GRPO
  recipe, ``robotwin_click_bell_grpo_lingbotvla.yaml``); True -> every step is an
  SDE step, an N(0, 1) initial term is prepended, and the per-step log-probs are
  averaged (mean over steps).

Weight sync (actor -> rollout): the actor tree is
``vla_model.model.qwenvl_with_expert.{qwenvl.model, qwenvl.visual,
qwen_expert.model}.*`` + ``vla_model.model.{state_proj, action_in_proj,
action_out_proj, action_time_mlp_in, action_time_mlp_out}.*`` (the checkpoint
keys, prefixed by ``vla_model.``). This class registers the vvla policy's stock
Qwen2.5-VL text model / ViT / vendored expert / projection heads under exactly
those key paths (a mirror ``nn.Module`` tree), so the state_dict key set matches
the actor natively -- the patch syncer's strict key-set equality holds and its
in-place ``copy_`` keeps the vvla forward (and any captured graph) valid. The
vvla ``LingBotVLAPolicy`` object is NOT registered (it shares every weight
module; registering it would duplicate the tree). The mirror tree's
``vla_model.model.config`` doubles as the config the inherited
``_preprocess_observation`` reads for ``prepare_state/language/images``.
"""

import math
import os
import random
from typing import Any, Literal

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.lingbotvla.lingbotvla_action_model import (
    LingbotvlaActionModel,
)


def _first_int(*vals, default):
    """First non-None value coerced to int, else ``default``.

    The RoboTwin ``lingbotvla_cli.yaml`` leaves ``action_dim: null``, so the native
    ``int(getattr(qwen_config, 'action_dim', ...))`` chain would hit ``int(None)``;
    coalescing None -> the RLinf cfg -> default keeps it robust to that."""
    for v in vals:
        if v is not None:
            return int(v)
    return int(default)


class VvlaLingbotvlaActionModel(LingbotvlaActionModel):
    """Rollout-only LingBot-VLA policy served by the vvla engine.

    ``predict_action_batch`` (inherited) mirrors the native output contract
    field-for-field. Supported configuration: ``noise_method == 'flow_sde'``
    with the standard (no value / value head) recipe; ``flow_noise`` /
    ``flow_cps`` / ``use_dsrl`` are rejected at construction.
    """

    def __init__(self, config, torch_dtype=torch.bfloat16):
        # Skip LingbotvlaActionModel.__init__ (it builds + loads the 16 GB native
        # model); build only the lightweight config/normaliser/processor pieces
        # its observation pipeline needs, plus the vvla rollout policy.
        nn.Module.__init__(self)
        from rlinf.utils.logging import get_logger

        self.config = config
        self.torch_dtype = torch_dtype
        self.logger = get_logger()
        self.global_step = 0

        if getattr(config, "use_dsrl", False):
            raise ValueError("vvla lingbotvla rollout backend does not support use_dsrl")
        noise_method = getattr(config, "noise_method", "flow_sde")
        if noise_method != "flow_sde":
            raise ValueError(
                "vvla lingbotvla rollout backend supports noise_method='flow_sde' only, "
                f"got '{noise_method}'"
            )

        qwen_config = self._build_qwen_config(config)

        self.action_dim = int(qwen_config.action_dim)
        self.action_chunk = int(qwen_config.n_action_steps)
        self.action_env_dim = int(getattr(config, "action_env_dim", self.action_dim))
        self.num_steps = int(getattr(config, "num_steps", 10))
        self.noise_method = noise_method
        self.noise_level = float(getattr(config, "noise_level", 0.5))
        self.joint_logprob = bool(getattr(config, "joint_logprob", False))
        self.image_size = int(self._data_config.get("img_size", 224))
        self.max_action_dim = int(qwen_config.max_action_dim)

        # value / noise heads: the RoboTwin GRPO recipe is add_value_head=False;
        # reject the unsupported branches rather than silently return zeros wrong.
        if getattr(config, "add_value_head", False):
            raise ValueError(
                "vvla lingbotvla rollout backend currently supports add_value_head=False only"
            )

        # observation-pipeline dependencies (native lines 246-278): the Qwen
        # processor + the RoboTwin normaliser, built exactly as the actor does.
        from lingbotvla.data.vla_data.transform import Normalizer
        from lingbotvla.models import build_processor

        self.processor = build_processor(config.tokenizer_path)
        self.language_tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        self.normalizer = self._build_normalizer(config, Normalizer)

        # vvla rollout policy: stock Qwen2.5-VL VL backbone + vendored expert,
        # vvla-owned MoT denoise loop. Its initial weights come from a HF
        # safetensors dir (posttrain SFT / base); the actor's weights overwrite
        # them via weight sync before / between rollouts. The RL checkpoint
        # (FSDP full_weights.pt) is not a safetensors dir, so a separate
        # ``vvla.checkpoint`` (or config_path) is used for the initial load.
        vv = getattr(config, "vvla", {}) or {}
        vv = dict(vv) if not isinstance(vv, dict) else vv
        ckpt = vv.get("checkpoint") or self._config_path
        from vvla.policies.factory import make_policy

        # eager is the default: native's MoT shared attention is
        # ``our_eager_attention_forward`` (flash/fa2 is NotImplemented in the vendored
        # model), so vvla's eager backend matches it and the ratio-at-theta0 stays tight
        # (q01-q99 [0.987, 1.015]); sdpa drifts to [0.835, 1.16].
        policy = make_policy(
            "lingbot_vla", checkpoint=str(ckpt), attention=vv.get("attention", "eager")
        )

        # Register the shared weight modules under the ACTOR's key layout so the
        # state_dict keys match natively (patch syncer requires strict key-set
        # equality). The vvla policy object itself is deliberately NOT registered.
        vla_model = nn.Module()
        vla_model.model = nn.Module()
        m = vla_model.model
        m.qwenvl_with_expert = nn.Module()
        m.qwenvl_with_expert.qwenvl = nn.Module()
        m.qwenvl_with_expert.qwenvl.model = policy._vl
        m.qwenvl_with_expert.qwenvl.visual = policy._visual
        m.qwenvl_with_expert.qwen_expert = nn.Module()
        m.qwenvl_with_expert.qwen_expert.model = policy._expert
        m.state_proj = policy._proj.state_proj
        m.action_in_proj = policy._proj.action_in_proj
        m.action_out_proj = policy._proj.action_out_proj
        m.action_time_mlp_in = policy._proj.action_time_mlp_in
        m.action_time_mlp_out = policy._proj.action_time_mlp_out
        m.config = qwen_config  # inherited _preprocess_observation reads this
        # vvla builds an fp32 master; cast to the rollout dtype (bf16, matching the
        # native actor) so weight-sync copy_ is dtype-aligned and the forward is bf16.
        self.vla_model = vla_model.to(torch_dtype)
        object.__setattr__(self, "vvla_policy", policy)

    # ---- config / normaliser build (native __init__ pieces, no model) ---------
    def _build_qwen_config(self, config):
        """Replicate ``LingbotvlaActionModel.__init__`` config assembly (lines
        144-195): lerobot ``PreTrainedConfig`` + ``lingbotvla_cli.yaml`` overrides
        + Qwen text/vision config merge. Records ``_config_path`` / ``_data_config``
        for reuse."""
        from lerobot.configs.policies import PreTrainedConfig
        from transformers import AutoConfig

        lingbotvla_cfg = getattr(config, "lingbotvla", getattr(config, "lingbot", config))
        model_path = getattr(config, "model_path", None)
        config_path = getattr(lingbotvla_cfg, "config_path", None) or model_path
        if not config_path:
            raise ValueError(
                "vvla lingbotvla requires actor.model.model_path or "
                "actor.model.lingbotvla.config_path for RoboTwin SFT config loading."
            )
        self._config_path = config_path

        training_config = self._load_training_config(config_path, model_path)
        self._data_config = training_config.get("data", {})
        training_model_config = dict(training_config["model"])
        training_model_config.update(training_config["train"])

        qwen_config = PreTrainedConfig.from_pretrained(config_path)
        for key, value in training_model_config.items():
            setattr(qwen_config, key, value)
        qwen_config.attention_implementation = "eager"
        qwen_config.tokenizer_path = config.tokenizer_path
        qwen_config.loss_type = getattr(qwen_config, "loss_type", "L1_fm")
        qwen_config.align_params = getattr(qwen_config, "align_params", {})
        qwen_config.norm_qkv = getattr(qwen_config, "norm_qkv", False)
        qwen_config = self._merge_qwen_config(
            qwen_config, AutoConfig.from_pretrained(config.tokenizer_path)
        )
        if training_config["model"].get("vocab_size", 0) != 0:
            qwen_config.vocab_size = training_config["model"]["vocab_size"]
        qwen_config.action_dim = _first_int(
            getattr(qwen_config, "action_dim", None), getattr(config, "action_dim", None), default=14
        )
        qwen_config.n_action_steps = _first_int(
            getattr(qwen_config, "chunk_size", None),
            getattr(qwen_config, "n_action_steps", None),
            getattr(config, "num_action_chunks", None),
            default=50,
        )
        qwen_config.max_action_dim = _first_int(getattr(qwen_config, "max_action_dim", None), default=75)
        qwen_config.max_state_dim = _first_int(getattr(qwen_config, "max_state_dim", None), default=75)
        return qwen_config

    def _build_normalizer(self, config, Normalizer):
        """Native __init__ lines 250-278: load the RoboTwin norm stats + build the
        ``robotwin_rep`` normaliser (identity images, ``norm_type`` state/action)."""
        import json

        lingbotvla_cfg = getattr(config, "lingbotvla", getattr(config, "lingbot", config))
        stats_json_path = getattr(
            lingbotvla_cfg,
            "stats_path",
            os.path.join(
                os.environ.get("LINGBOT_VLA_PATH", ""),
                "assets/norm_stats/robotwin_all_new.json",
            ),
        )
        if not os.path.exists(stats_json_path):
            raise FileNotFoundError(
                f"vvla lingbotvla RoboTwin SFT stats file not found: {stats_json_path} "
                "(set LINGBOT_VLA_PATH or actor.model.lingbotvla.stats_path)"
            )
        with open(stats_json_path) as f:
            raw_stats = json.load(f)
        self.norm_stats = raw_stats.get("norm_stats", raw_stats.get("stats", raw_stats))
        norm_type = self._data_config.get("norm_type", "bounds_99_woclip")
        return Normalizer(
            norm_stats=self.norm_stats,
            from_file=True,
            data_type="robotwin_rep",
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": norm_type,
                "action": norm_type,
            },
        )

    # ---- rollout sampling (vvla forward + flow-SDE kernel) --------------------
    def _encode_prefix_vvla(self, images, img_masks, lang_tokens, lang_masks, state):
        """Build a vvla ``LingBotVLABatch`` from the native-preprocessed tensors and
        encode the VL prefix once.

        ``images`` is the native ``_preprocess_observation`` output: a list whose
        first element is ``[B, n_cam, n_patch, 1176]`` Qwen pixel values. Qwen packs
        every image's patches along dim 0, so we flatten to ``[B*n_cam*n_patch, 1176]``
        with one ``grid_thw`` row per camera-image and let vvla's ``_embed_image``
        reshape back by the observation count (``batch_size``)."""
        from vvla.policies.lingbot_vla.processor_lingbot_vla import LingBotVLABatch

        device = next(self.parameters()).device
        vla_images = images[0] if isinstance(images, list) else images
        B, n_cam, n_patch, p_dim = vla_images.shape
        pixel_values = vla_images.reshape(B * n_cam * n_patch, p_dim).to(device, self.torch_dtype)
        hw = int(round(n_patch**0.5))
        grid = torch.tensor([[1, hw, hw]] * (B * n_cam), device=device)
        merged_per_cam = n_patch // 4  # Qwen 2x2 spatial merge
        img_pad = torch.ones(B, n_cam * merged_per_cam, device=device)
        batch = LingBotVLABatch(
            pixel_values=pixel_values,
            image_grid_thw=grid,
            img_pad_masks=img_pad,
            lang_tokens=lang_tokens.to(device),
            lang_pad_masks=lang_masks.to(device),
            state=state.to(device, self.torch_dtype),
            request_ids=[f"r{i}" for i in range(B)],
        )
        return self.vvla_policy.encode_prefix(batch)

    def _sample_mean_var(self, x_t, idx, prefix, timesteps, noise_level, mode):
        """One flow-SDE transition mean/std, replicating ``sample_mean_var_val``'s
        flow_sde branch verbatim, with ``v_t`` from the vvla denoise step.

        Same tensor-expression order and dtype promotion as the native head (the
        fp32 ``noise_level`` / ``timesteps`` scalars promote the products), so the
        value is bit-identical to what the actor recomputes on the same chains."""
        bsize = x_t.shape[0]
        device = x_t.device
        idx_b = torch.full((bsize,), idx, device=device, dtype=torch.long)
        t_input = timesteps[idx_b]
        delta = timesteps[idx_b] - timesteps[idx_b + 1]

        # vvla velocity field: v(x_t, t | prefix) -> [B, chunk, max_action_dim].
        # The forward runs in the model dtype (native casts ``x_t.to(torch_dtype)``
        # inside get_suffix_out); the mean math below keeps the original-dtype x_t.
        t_vec = torch.full((bsize,), float(timesteps[idx]), device=device, dtype=torch.float32)
        v_t = self.vvla_policy.denoise_step(x_t.to(self.torch_dtype), t_vec, prefix)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        else:  # train, flow_sde
            sigmas = (
                noise_level
                * torch.sqrt(
                    timesteps / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                )[:-1]
            )
            sigma_i = sigmas[idx_b][:, None, None].expand_as(x_t)
            x0_weight = torch.ones_like(t_input) - (t_input - delta)
            x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
            x_t_std = torch.sqrt(delta) * sigma_i
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    @torch.no_grad()
    def sample_actions(self, observation, noise=None, mode="train", compute_values=True):
        """vvla rollout mirroring native ``sample_actions``'s output contract:
        ``{actions, chains, prev_logprobs, prev_values, denoise_inds, lang_tokens,
        lang_masks}``. The inherited ``predict_action_batch`` consumes it unchanged."""
        del compute_values  # add_value_head=False -> no value computed
        bsize = (
            observation.state.shape[0]
            if hasattr(observation.state, "shape")
            else len(observation.state)
        )
        device = next(self.parameters()).device
        num_steps = self.num_steps
        max_act_dim = self.max_action_dim
        horizon = int(getattr(self.config, "action_horizon", self.action_chunk))

        if noise is None:
            # native ``sample_noise`` = ``torch.randn(shape, device)`` -> fp32 chains
            noise = torch.randn(bsize, horizon, max_act_dim, device=device)
        else:
            noise = noise.to(device=device, dtype=self.torch_dtype)
            if noise.shape[-1] < max_act_dim:
                pad = torch.randn(
                    (*noise.shape[:-1], max_act_dim - noise.shape[-1]),
                    device=device,
                    dtype=self.torch_dtype,
                )
                noise = torch.cat([noise, pad], dim=-1)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )
        state = self._reorder_rep_state_for_model(state)
        prefix = self._encode_prefix_vvla(images, img_masks, lang_tokens, lang_masks, state)

        timesteps = torch.linspace(1, 1 / num_steps, num_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        noise_level = torch.tensor(self.noise_level).to(device)

        x_t = noise
        chains = [x_t]
        log_probs = []
        if self.joint_logprob:
            log_probs.append(
                self.get_logprob_norm(x_t, torch.zeros_like(noise), torch.ones_like(noise))
            )
            denoise_inds = torch.arange(num_steps, device=device)
        elif mode == "train":
            if getattr(self.config, "ignore_last", False):
                sel = random.randint(0, num_steps - 2)
            else:
                sel = random.randint(0, num_steps - 1)
            denoise_inds = torch.tensor([sel] * num_steps, device=device)
        else:
            denoise_inds = torch.tensor([-1] * num_steps, device=device)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        for idx in range(num_steps):
            sample_mode = "train" if idx == int(denoise_inds[0][idx]) else "eval"
            x_t_mean, x_t_std = self._sample_mean_var(
                x_t, idx, prefix, timesteps, noise_level, sample_mode
            )
            x_t = x_t_mean + torch.randn(x_t.shape, device=device) * x_t_std
            log_probs.append(self.get_logprob_norm(x_t, x_t_mean, x_t_std))
            chains.append(x_t)

        x_0 = x_t
        env_actions = self._select_env_action_dims(x_0[:, : self.action_chunk, :])
        chains = torch.stack(chains, dim=1)
        log_probs = self._select_env_action_dims(
            torch.stack(log_probs, dim=1)[:, :, : self.action_chunk, :]
        )
        if self.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[torch.arange(log_probs.shape[0]), denoise_inds[:, 0]]

        values = torch.zeros((bsize, 1), device=device, dtype=self.torch_dtype)
        return {
            "actions": env_actions,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }

    # ---- BasePolicy interface -------------------------------------------------
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "vvla_lingbotvla is a rollout backend; keep actor.model.model_type=lingbotvla"
        )


def get_model(cfg, torch_dtype=None):
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    model = VvlaLingbotvlaActionModel(cfg, torch_dtype=torch_dtype)
    return model.eval()
