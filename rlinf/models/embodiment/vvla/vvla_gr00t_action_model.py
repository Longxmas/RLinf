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

"""vvla-backed GR00T N1.7 rollout policy.

Design (rollout-side swap, mirroring the pi0.5 vvla adapter):
  * the FSDP actor keeps the reference GR00T implementation
    (``GR00T_N1_7_ForRLActionPrediction``) — training, ``default_forward``
    logprob recompute, and the value head are unchanged;
  * this class serves *rollout generation only*: it wraps a vvla
    ``Gr00tPolicy`` (vendored action head + stock Qwen3-VL backbone,
    vvla-owned denoise loop) and produces ``forward_inputs`` in exactly the
    RLinf GR00T convention (``chains`` / ``denoise_inds`` / stashed processor
    outputs) so the actor consumes them unchanged.

Sampling (SDE loop owned by the vvla engine, this class is format glue):
  * the denoise loop is ``vvla.rollout.logprob.flow_sample_with_logprob``
    driven with GR00T's transition kernel. GR00T integrates FORWARD time
    (t: 0 -> 1, x0 noise -> x1 action; cf. pi0.5's 1 -> 0), so the hooks are
    the mirror image of the openpi ones::

        sigma(t)  = noise_level * sqrt((1 - t) / t),  t == 0 clamped to 1/N
        mean      = x + v*dt - (sigma(t)^2 * dt / (2*(1 - t))) * (x - v*t)
        std       = sigma(t) * sqrt(dt)

    verified algebraically identical to ``sample_mean_var_val``'s
    x0/x1-weight form (the correction acts on ``x0_pred = x - v*t``; the
    plain part collapses to Euler ``x + v*dt``). Noise is injected on the one
    randomly selected denoise step only; all other steps are deterministic.
  * ``prev_logprobs`` follows the GR00T convention: the FULL per-step
    elementwise log-probability tensor ``[B, N, action_chunk,
    env_action_dim]`` (zeros on the deterministic steps), from which the
    actor's ``default_forward`` gathers the selected step. Computed from the
    sampler-recorded per-step velocities — no extra denoise forward.

Weight sync (actor -> rollout): the actor tree is ``backbone.model.<qwen3vl>``
+ ``action_head.<head>`` (+ ``action_head.value_head``). This class registers
the vvla policy's stock Qwen3-VL backbone under ``backbone.model`` and the
vendored action head under ``action_head`` (attaching a ``value_head`` of the
actor's exact shape), so the state_dict keys match the actor's natively — no
remap. The vvla ``Gr00tPolicy`` object itself is NOT registered (it shares
every weight module; registering it would duplicate the tree and break the
patch syncer's strict key-set equality). vvla keeps the backbone's vestigial
final text norm registered (bypassed in forward) so the key set stays
checkpoint-complete.

Input pipeline: the GR00T obs conversion + ``Gr00tN1d7Processor`` transforms
(including the deliberate state bf16 round-trip and the right-padding to
``padding_value``) are reused verbatim via unbound-method delegation, so the
processor outputs are bit-identical to the actor's. Action decode
(unnormalize, relative EEF -> absolute, chunking, exploration noise) is the
same shared code.
"""

import math
import random
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    GR00T_N1_7_ForRLActionPrediction,
    _canonicalize_gr00t_text_forward_inputs,
    _find_processor_dir,
    _normalize_gr00t_forward_inputs,
    _resolve_env_action_dim,
    redirect_qwen3_backbone_to_local,
)
from rlinf.models.embodiment.gr00t.simulation_io import (
    ACTION_CONVERSION_N1D7,
    OBS_CONVERSION,
)
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger


def _gr00t_flow_sde_sigma(t_val: float, num_steps: int, noise_level: float) -> float:
    """GR00T's flow-SDE sigma schedule as a scalar function of t.

    Reference (``sample_mean_var_val``, flow_sde branch)::

        timesteps = linspace(0, 1, N + 1)                # t_k = k/N
        sigma(t_k) = noise_level * sqrt((1 - t_k) / where(t_k == 0, t_1, t_k))

    i.e. sigma(t) = noise_level*sqrt((1-t)/t) with the t == 0 grid endpoint's
    denominator clamped to t_1 = 1/N. The grid coincides exactly with vvla's
    default ``flow_schedule`` (t from 0 -> 1, dt = +1/N)."""
    denom = t_val if t_val > 0.0 else 1.0 / num_steps
    return noise_level * math.sqrt((1.0 - t_val) / denom)


def _gr00t_flow_sde_mean(
    x: torch.Tensor, v: torch.Tensor, t_val: float, dt: float, sig: float
) -> torch.Tensor:
    """GR00T flow_sde transition mean, as a vvla sampler ``mean_fn``.

    Algebraically identical to ``sample_mean_var_val``'s x0/x1-weight form:
    the sigma-correction subtracts ``(sigma^2*dt/(2*(1-t))) * x0_pred`` with
    ``x0_pred = x - v*t``; on sigma == 0 (deterministic) steps this reduces to
    the plain Euler mean ``x + v*dt``."""
    mean = x + v * dt
    if sig > 0.0:
        mean = mean - (sig * sig * dt / (2.0 * (1.0 - t_val))) * (x - v * t_val)
    return mean


class VvlaGr00tForRLActionPrediction(nn.Module, BasePolicy):
    """Rollout-only GR00T N1.7 policy served by the vvla engine.

    ``predict_action_batch`` mirrors ``GR00T_N1_7_ForRLActionPrediction``'s
    output contract field-for-field. Supported configuration:
    ``noise_method == 'flow_sde'`` with the standard value head (the RLinf
    GR00T PPO recipe); joint_logprob / reinflow are rejected at construction.
    """

    def __init__(self, cfg):
        super().__init__()
        vv = cfg.get("vvla", {}) or {}
        self.logger = get_logger()

        # ---- config (same sources as the native builder) ----------------------
        # Same embodiment-tag patch the native gr00t_n1d7 builder applies (the
        # RLinf enum carries the sim tags the upstream one lacks).
        from rlinf.utils.patcher import Patcher

        Patcher.clear()
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EmbodimentTag",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
        )
        Patcher.apply()
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
        from gr00t.data.embodiment_tags import EmbodimentTag

        self.config = Gr00tN1d7Config.from_pretrained(str(cfg.model_path))
        rl_cfg = dict(cfg.rl_head_config)
        rl_cfg.setdefault("noise_method", "flow_sde")
        rl_cfg.setdefault("noise_level", 0.5)
        rl_cfg.setdefault("noise_anneal", False)
        self.rl_config = rl_cfg

        if rl_cfg.get("noise_method") != "flow_sde":
            raise ValueError(
                "vvla gr00t rollout backend supports noise_method='flow_sde' only, "
                f"got '{rl_cfg.get('noise_method')}'"
            )
        if rl_cfg.get("joint_logprob"):
            raise ValueError("vvla gr00t rollout backend does not support joint_logprob")

        self.embodiment_tag = EmbodimentTag(cfg.embodiment_tag)
        self.padding_value = rl_cfg.get("padding_value", 0)
        self.compute_dtype = torch.bfloat16
        self.output_action_chunks = int(cfg.get("num_action_chunks", 1))
        self.obs_converter_type = cfg.get("obs_converter_type", "libero")
        self.obs_convert_fn = OBS_CONVERSION[self.obs_converter_type]
        self.action_convert_fn = ACTION_CONVERSION_N1D7[self.obs_converter_type]
        self.num_steps = int(
            cfg.get("denoising_steps", getattr(self.config, "num_inference_timesteps", 4))
        )
        self.num_timestep_buckets = getattr(self.config, "num_timestep_buckets", 1000)
        self.model_action_dim = getattr(
            self.config, "max_action_dim", getattr(self.config, "action_dim", 7)
        )
        self.action_horizon = getattr(self.config, "action_horizon", 16)

        # ---- processor (actor-identical, offline via the backbone redirect) ---
        backbone_model_path = str(cfg.backbone_model_path)
        processor_dir = _find_processor_dir(Path(cfg.model_path))
        if processor_dir is None:
            raise FileNotFoundError(f"no GR00T processor files under {cfg.model_path}")
        with redirect_qwen3_backbone_to_local(
            str(self.config.model_name), backbone_model_path
        ):
            self._modality_transform, self._modality_config = (
                GR00T_N1_7_ForRLActionPrediction._load_processor_from_dir(
                    processor_dir, backbone_model_path=backbone_model_path
                )
            )
        # metadata -> valid_action_dim / image_nums (same fallback chain)
        exp_cfg_path = Path(cfg.model_path) / "experiment_cfg"
        GR00T_N1_7_ForRLActionPrediction._load_metadata(self, exp_cfg_path)
        self.action_dim = _resolve_env_action_dim(
            cfg.get("action_dim", None), self.valid_action_dim
        )
        self.env_action_dim = self.action_dim
        self.action_chunk = self.output_action_chunks

        # ---- vvla policy: vendored head + stock Qwen3-VL, vvla-owned loop -----
        from vvla.policies.factory import make_policy

        policy = make_policy(
            "gr00t",
            checkpoint=str(cfg.model_path),
            cosmos_path=backbone_model_path,
            attention=vv.get("attention", "sdpa"),
        )
        # Register the shared weight modules under the ACTOR's key layout:
        #   backbone.model.<qwen3vl...>  /  action_head.<head...>
        # so state_dict keys match the actor natively (patch syncer requires
        # strict key-set equality). The policy object itself is deliberately
        # NOT registered (it would duplicate the whole tree).
        self.backbone = nn.Module()
        self.backbone.model = policy._backbone
        self.action_head = policy._ah
        object.__setattr__(self, "vvla_policy", policy)

        if rl_cfg.get("add_value_head", False):
            vlm_width = getattr(self.config, "backbone_embedding_dim", 2048)
            state_width = getattr(self.config, "input_embedding_dim", 1536)
            proj_width = (
                vlm_width if rl_cfg.get("use_vlm_value", False) else vlm_width + state_width
            )
            self.action_head.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
            # match the native builder: bf16 like the rest of the head, then init
            self.action_head.value_head.to(torch.bfloat16)
            self.action_head.value_head._init_weights()

        self.global_step = 0

    # ---- reused native helpers (unbound-method delegation) --------------------
    def apply_transforms(self, obs):
        return GR00T_N1_7_ForRLActionPrediction.apply_transforms(self, obs)

    def unapply_transforms(self, action, state=None):
        return GR00T_N1_7_ForRLActionPrediction.unapply_transforms(self, action, state)

    # staticmethods the delegated pipeline resolves via self
    _check_state_is_batched = staticmethod(
        GR00T_N1_7_ForRLActionPrediction._check_state_is_batched
    )
    _coerce_observation_values_to_numpy = staticmethod(
        GR00T_N1_7_ForRLActionPrediction._coerce_observation_values_to_numpy
    )

    def _prepare_rollout_observation(self, env_obs):
        return GR00T_N1_7_ForRLActionPrediction._prepare_rollout_observation(
            self, env_obs
        )

    def _get_unnormalized_action(self, normalized_action, state=None):
        return GR00T_N1_7_ForRLActionPrediction._get_unnormalized_action(
            self, normalized_action, state
        )

    def _apply_exploration_noise(self, raw_action, mode):
        # native reads self.action_head.rl_config; provide it transiently-free
        # by inlining the same 8 lines against self.rl_config.
        if mode != "train":
            return raw_action
        noise_scale = float(self.rl_config.get("action_noise_scale", 0.1))
        if noise_scale <= 0:
            return raw_action
        import numpy as np

        is_numpy = isinstance(raw_action, np.ndarray)
        raw_tensor = torch.from_numpy(raw_action) if is_numpy else raw_action
        noise = torch.randn_like(raw_tensor) * noise_scale
        raw_tensor = (raw_tensor + noise).clamp(-1.0, 1.0)
        return raw_tensor.numpy() if is_numpy else raw_tensor

    def _finalize_rollout_forward_inputs(self, forward_inputs):
        return GR00T_N1_7_ForRLActionPrediction._finalize_rollout_forward_inputs(
            self, forward_inputs
        )

    def get_logprob_norm(self, sample, mu, sigma):
        """Elementwise Gaussian log-prob, exactly the native head's convention
        (sigma == 0 coordinates contribute zero)."""
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    def set_global_step(self, global_step):
        self.global_step = global_step

    # ---- flow-SDE sampling via the vvla engine --------------------------------
    def _make_sigma_schedule(self, sde_step, num_steps, noise_level):
        t_grid = [t for t, _ in self.vvla_policy.flow_schedule(num_steps)]
        t_sde = t_grid[sde_step] if sde_step is not None else None
        tol = 0.25 / num_steps

        def sigma(t_val: float) -> float:
            if t_sde is None or abs(t_val - t_sde) > tol:
                return 0.0
            return _gr00t_flow_sde_sigma(t_val, num_steps, noise_level)

        return sigma

    def _per_step_logprobs(self, chains, velocities, sde_step, num_steps, noise_level):
        """GR00T-convention ``prev_logprobs``: [B, N, action_chunk, env_dim],
        elementwise, zeros on the deterministic steps — from the recorded
        chains/velocities, no extra forward.

        The selected step's mean/std are computed with the native
        ``sample_mean_var_val`` tensor expressions VERBATIM (x0/x1-weight form,
        same dtype-promotion order), so the value is bit-identical to what the
        actor's recompute produces on the same chains — the PPO ratio at
        theta == theta_behavior carries no formula-rounding noise."""
        device, dtype = chains.device, chains.dtype
        bsize = chains.shape[0]
        steps = []
        # native: timesteps/sigmas built in the model compute dtype; noise_level
        # as an fp32 scalar tensor (the products then promote to fp32).
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
        noise_t = torch.tensor(noise_level).to(device)
        sigmas = noise_t * torch.sqrt(
            (1 - timesteps) / torch.where(timesteps == 0, timesteps[1], timesteps)
        )[:-1]
        for k in range(num_steps):
            x_t = chains[:, k]
            if k != sde_step:
                steps.append(torch.zeros_like(x_t, dtype=torch.float32))
                continue
            v_t = velocities[:, k]
            idx = torch.full((bsize,), k, dtype=torch.long, device=device)
            t_input = timesteps[idx][:, None, None].expand_as(x_t)
            delta = (timesteps[idx + 1] - timesteps[idx])[:, None, None].expand_as(x_t)
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            x0_pred = x_t - v_t * t_input
            x1_pred = x_t + v_t * (1 - t_input)
            x0_weight = (
                torch.ones_like(t_input)
                - (t_input + delta)
                - sigma_i**2 * delta / (2 * (1 - t_input))
            )
            x1_weight = t_input + delta
            x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
            x_t_std = torch.sqrt(delta) * sigma_i
            steps.append(self.get_logprob_norm(chains[:, k + 1], x_t_mean, x_t_std))
        return torch.stack(steps, dim=1)[
            :, :, : self.action_chunk, : self.env_action_dim
        ]

    @torch.no_grad()
    def _sample_actions_vvla(self, normalized_input, mode="train"):
        from vvla.policies.gr00t.processor_gr00t import Gr00tBatch
        from vvla.rollout.logprob import flow_sample_with_logprob

        bi = {
            k: normalized_input[k]
            for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
            if k in normalized_input
        }
        batch = Gr00tBatch(
            backbone_inputs=bi,
            state=normalized_input["state"],
            embodiment_id=normalized_input["embodiment_id"].reshape(-1).long(),
        )
        # The processor emits CPU tensors (the native backbone wrapper moves
        # them internally); vvla's encode_prefix expects device-resident inputs.
        device = next(self.parameters()).device
        batch = batch.to(device)
        prefix = self.vvla_policy.encode_prefix(batch)
        bsize = prefix.batch_size
        num_steps = self.num_steps

        # noise in the model compute dtype (native: x_t = randn(..., vl.dtype))
        x0 = torch.randn(
            bsize,
            self.action_horizon,
            self.model_action_dim,
            device=device,
            dtype=prefix.vl_embeds.dtype,
        )

        if mode == "train":
            sde_step = random.randint(0, num_steps - 1)
            denoise_inds = torch.full(
                (bsize, num_steps), sde_step, dtype=torch.long, device=device
            )
        else:
            sde_step = None
            denoise_inds = torch.full(
                (bsize, num_steps), -1, dtype=torch.long, device=device
            )

        noise_level = float(self.rl_config.get("noise_level", 0.5))
        sigma_fn = self._make_sigma_schedule(sde_step, num_steps, noise_level)
        actions, _lp, chains, velocities = flow_sample_with_logprob(
            self.vvla_policy,
            prefix,
            x0,
            num_steps,
            sigma=sigma_fn,
            return_trajectory=True,
            per_step=True,
            mean_fn=_gr00t_flow_sde_mean,
            return_velocities=True,
        )

        prev_logprobs = self._per_step_logprobs(
            chains, velocities, sde_step, num_steps, noise_level
        )

        if self.rl_config.get("add_value_head", False):
            vl_pooled = prefix.vl_embeds.mean(dim=1)
            if self.rl_config.get("use_vlm_value", False):
                value_embs = vl_pooled
            else:
                value_embs = torch.cat(
                    (vl_pooled, prefix.state_features.reshape(bsize, -1)), dim=1
                )
            values = self.action_head.value_head(value_embs)[:, 0][:, None]
        else:
            values = torch.zeros(bsize, 1, device=device, dtype=prefix.vl_embeds.dtype)

        return {
            "actions": actions,
            "chains": chains,
            "prev_logprobs": prev_logprobs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    # ---- BasePolicy interface --------------------------------------------------
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "vvla_gr00t is a rollout backend; keep actor.model.model_type=gr00t_n1d7"
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ):
        """GR00T-contract rollout step: env obs -> (env actions, RL extras)."""
        del kwargs
        observations, obs_copy, is_batch = self._prepare_rollout_observation(env_obs)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_input = (
            GR00T_N1_7_ForRLActionPrediction._cast_float_tensors_to_compute_dtype(
                normalized_input, self.compute_dtype
            )
        )
        normalized_input = _canonicalize_gr00t_text_forward_inputs(
            normalized_input, self.padding_value
        )

        if mode == "eval":
            normalized_input = _normalize_gr00t_forward_inputs(normalized_input)
            outputs = self._sample_actions_vvla(normalized_input, mode="eval")
            normalized_action = outputs["actions"].float()
            result: dict[str, Any] = {
                "prev_logprobs": None,
                "prev_values": None,
                "forward_inputs": {},
            }
        else:
            normalized_input = _normalize_gr00t_forward_inputs(normalized_input)
            outputs = self._sample_actions_vvla(normalized_input, mode="train")
            normalized_action = outputs["actions"].float()
            batch_size = normalized_action.shape[0]
            from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
                _batchify_gr00t_forward_input,
            )

            stashed = {
                key: _batchify_gr00t_forward_input(key, value, batch_size)
                for key, value in normalized_input.items()
            }
            forward_inputs = {
                "chains": outputs["chains"],
                "denoise_inds": outputs["denoise_inds"],
                **stashed,
            }
            result = {
                "prev_logprobs": outputs["prev_logprobs"],
                "prev_values": outputs["prev_values"],
                "forward_inputs": self._finalize_rollout_forward_inputs(forward_inputs),
            }

        unnormalized_action = self._get_unnormalized_action(
            normalized_action, state=observations
        )
        if not is_batch:
            from rlinf.models.embodiment.gr00t.utils import squeeze_dict_values

            unnormalized_action = squeeze_dict_values(unnormalized_action)
        raw_action = self.action_convert_fn(
            unnormalized_action, chunk_size=self.output_action_chunks
        )
        raw_action = self._apply_exploration_noise(raw_action, mode)
        return raw_action, result


def get_model(cfg, torch_dtype=None):
    # NOTE: no global matmul-precision change here — the native GR00T rollout
    # process leaves torch defaults untouched (the model runs bf16 end to end),
    # and the adapter mirrors that regime.
    model = VvlaGr00tForRLActionPrediction(cfg)
    return model.eval()
