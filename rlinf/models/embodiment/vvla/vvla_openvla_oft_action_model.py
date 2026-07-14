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

"""vvla-backed OpenVLA-OFT rollout policy.

Design (rollout-side swap, mirroring the pi0.5 / GR00T vvla adapters):
  * the FSDP actor keeps the reference implementation
    (``OpenVLAOFTForRLActionPrediction``, prismatic modeling) — training,
    ``default_forward`` logprob recompute, and the (optional) value head are
    unchanged;
  * this class serves *rollout generation only*: it wraps a vvla
    ``OpenVLAOFTPolicy`` (vvla-owned Llama-2 forward over the checkpoint's
    vision_backbone / projector / language_model weights, single-pass
    categorical action-token decode) and produces ``forward_inputs`` in exactly
    the native OFT convention (``input_ids`` / ``attention_mask`` /
    ``pixel_values`` / ``action_tokens``) so the actor consumes them unchanged.

Sampling (single causal decode owned by the vvla engine, this class is format
glue): the 56 action tokens are produced by one ``encode_prefix`` +
``decode_action_logits`` + ``head.sample`` — no time schedule, no noise seed.
The head replicates op-for-op RLinf's sampling / log-prob / detokenisation
(mask non-bin logits -> ``/temperature`` -> top-k -> softmax -> multinomial;
``prev_logprobs`` = token-level ``-cross_entropy`` on the temp-scaled, top-k,
bin-masked logits at the sampled ids), so a vvla rollout is bit-aligned against
the native generator (verified: vision max|Δ|=0, action-token argmax 56/56,
action-logits max|Δ|≈1.6e-4 cross-implementation fp32 noise).

Weight sync (actor -> rollout): the native actor tree (prismatic
``OpenVLAForActionPrediction``) is ``vision_backbone.* / projector.* /
language_model.*`` (+ ``value_head.*``). This class registers the vvla policy's
same weight modules under those exact names, so ``state_dict()`` keys match the
actor's natively — NO remap, no state_dict override. The vvla ``OpenVLAOFTPolicy``
object itself is NOT registered as a submodule (it shares every weight module;
registering it would duplicate the whole tree under ``vvla_policy.*`` and break
the patch syncer's strict key-set equality). Device / dtype moves on the
registered modules propagate to the vvla policy automatically (shared Parameters).

Input pipeline: the native OFT ``MultiInputPrismaticProcessor`` (the actor's own
``input_processor``) is reused verbatim to turn env obs into ``input_ids`` /
``pixel_values``, so the tokenized prompt and 6-channel [DINOv2 ‖ SigLIP] pixels
are bit-identical to the actor's; those tensors are fed straight into a vvla
``OpenVLAOFTBatch`` (vvla's own ``OpenVLAOFTProcessor.collate`` is bypassed — it
tokenizes at ``max_length=50`` and would diverge from the actor's 128).
"""

import numpy as np
import torch
from torch import nn

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.utils.logging import get_logger

# The env-obs -> (input_ids, attention_mask, pixel_values) preamble below is a
# verbatim port of ``OpenVLAOFTForRLActionPrediction.predict_action_batch``'s
# first half; it MUST stay in sync with that method (same prompt template, image
# permute, input_processor call, [B, N, C, H, W] -> [B, N*C, H, W] reshape).
_PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut: "
_SPACE_TOKEN = 29871


class VvlaOpenVLAOFTForRLActionPrediction(nn.Module, BasePolicy):
    """Rollout-only OpenVLA-OFT policy served by the vvla engine.

    ``predict_action_batch`` mirrors ``OpenVLAOFTForRLActionPrediction``'s output
    contract field-for-field: returns ``(chunk_actions [B, num_chunks,
    action_dim], {prev_logprobs [B, 56], prev_values [B, 1], forward_inputs})``
    with ``forward_inputs = {input_ids [B, L(=max_prompt_length), last=space],
    attention_mask, pixel_values [B, N*C, H, W], action_tokens [B, num_chunks,
    action_dim]}``. The actor's ``default_forward`` consumes them unchanged.
    """

    def __init__(self, cfg):
        super().__init__()
        vv = cfg.get("vvla", {}) or {}
        self.logger = get_logger()

        # ---- RL geometry (single source of truth with env/worker/actor) -----
        self.action_dim = int(cfg.action_dim)
        self.num_action_chunks = int(cfg.num_action_chunks)
        self.n_tokens = self.action_dim * self.num_action_chunks
        self.max_prompt_length = int(cfg.max_prompt_length)
        self.num_images_in_input = int(cfg.get("num_images_in_input", 1))
        self.unnorm_key = cfg.unnorm_key
        self.value_type = cfg.get("value_type", "action_level")
        self.compute_dtype = torch_dtype_from_precision(cfg.get("precision", "bf16"))

        # ---- native config + input processor (actor-identical obs pipeline) --
        # Reuses the exact construction in the native OFT builder
        # (get_model_config_and_input_processor): OpenVLAConfig + the RLinf
        # MultiInputPrismaticProcessor, so input_ids / pixel_values are
        # bit-identical to the actor's.
        from rlinf.models.embodiment.openvla_oft.rlinf import (
            get_model_config_and_input_processor,
        )

        self._model_config, self.input_processor = (
            get_model_config_and_input_processor(cfg)
        )

        # ---- vvla policy: self-hosted forward over the ckpt weights ----------
        from vvla.policies.factory import make_policy

        policy = make_policy(
            "openvla_oft",
            checkpoint=str(cfg.model_path),
            attention=vv.get("attention", "eager"),
        )

        # Align the detok statistics with the actor's unnorm_key exactly (the
        # vvla builder picks unnorm_key from the ckpt config; the RLinf yaml is
        # the authority). q01/q99/mask drive _unnormalize_actions; bin_centers
        # are fixed. norm_stats comes from the native config (ckpt
        # dataset_statistics merged in by the builder).
        stats = self._resolve_action_stats()
        head = policy.head
        head.q01.copy_(torch.tensor(np.asarray(stats["q01"]), dtype=head.q01.dtype))
        head.q99.copy_(torch.tensor(np.asarray(stats["q99"]), dtype=head.q99.dtype))
        _mask = stats.get("mask", [True] * (self.action_dim - 1) + [False])
        head.norm_mask.copy_(torch.tensor(np.asarray(_mask), dtype=torch.bool))

        # ---- register shared weight modules under the ACTOR's key layout -----
        #   vision_backbone.*  /  projector.*  /  language_model.*
        # so state_dict keys equal the actor's natively (patch syncer requires
        # strict key-set equality). The vvla policy object is NOT registered.
        self.vision_backbone = policy.vision_backbone
        self.projector = policy.projector
        self.language_model = policy.language_model
        object.__setattr__(self, "vvla_policy", policy)

        # ---- optional value head (actor-shape; hidden = space position) ------
        # LIBERO-spatial GRPO uses add_value_head=False (GRPO group baseline),
        # so this is skipped there; kept for configs that enable it, matching
        # the native ValueHead spec so the weight-sync key set still matches.
        self.add_value_head = bool(cfg.get("add_value_head", False))
        if self.add_value_head:
            from rlinf.models.embodiment.modules.value_head import ValueHead

            hidden_size = int(cfg.get("hidden_size", self._model_config.hidden_size))
            output_dim = 1 if self.value_type == "chunk_level" else self.num_action_chunks
            self.value_head = ValueHead(
                input_dim=hidden_size,
                hidden_sizes=(512, 128),
                output_dim=output_dim,
                activation="gelu",
                bias_last=False,
            )

        # ---- precision regime: bf16 end-to-end, as the native OFT rollout ----
        # The registered modules (shared with vvla_policy) cast together; the
        # head's fp32 detok buffers live inside vvla_policy (unregistered) and
        # are untouched. vvla's self-hosted forward keeps RMSNorm variance in
        # fp32 internally regardless.
        self.to(self.compute_dtype)

        self.global_step = 0

    # ---- helpers --------------------------------------------------------------
    def _resolve_action_stats(self) -> dict:
        norm_stats = getattr(self._model_config, "norm_stats", {}) or {}
        key = self.unnorm_key
        if key not in norm_stats and f"{key}_no_noops" in norm_stats:
            key = f"{key}_no_noops"
        if key not in norm_stats:
            raise KeyError(
                f"unnorm_key '{self.unnorm_key}' not in norm_stats "
                f"(have {list(norm_stats)[:6]} ...)"
            )
        return norm_stats[key]["action"]

    def set_global_step(self, global_step):
        self.global_step = global_step

    def _apply(self, fn, *args, **kwargs):
        # nn.Module.to / cuda / cpu / half / float all route through _apply. The
        # shared weight modules (vision_backbone / projector / language_model) are
        # registered and move with super()._apply; the vvla policy's head is NOT
        # registered (object.__setattr__), so bring its fp32 detok buffers to the
        # same DEVICE here — device-only (never apply fn, which would also cast the
        # dtype and corrupt bin_centers / q01 / q99).
        super()._apply(fn, *args, **kwargs)
        if hasattr(self, "vvla_policy"):
            dev = next(self.language_model.parameters()).device
            self.vvla_policy.head.to(dev)
        return self

    # ---- env obs -> (input_ids, attention_mask, pixel_values) -----------------
    def _encode_obs(self, env_obs):
        """Verbatim port of the native predict_action_batch preamble."""
        task_descriptions = [
            _PROMPT_TEMPLATE.format(task=t.lower()) for t in env_obs["task_descriptions"]
        ]
        if env_obs["main_images"].ndim == 4:
            env_obs["main_images"] = env_obs["main_images"].unsqueeze(1)
        assert env_obs["main_images"].ndim == 5
        all_images = [env_obs["main_images"].permute(0, 1, 4, 2, 3)]  # [B,1,C,H,W]
        if self.num_images_in_input > 1:
            if env_obs["wrist_images"].ndim == 4:
                env_obs["wrist_images"] = env_obs["wrist_images"].unsqueeze(1)
            assert env_obs["wrist_images"].ndim == 5
            wrist_imgs = env_obs["wrist_images"].permute(0, 1, 4, 2, 3)
            all_images.extend([wrist_imgs[:, i] for i in range(wrist_imgs.shape[1])])

        device = next(self.parameters()).device
        precision = next(self.parameters()).dtype

        primary_image = all_images.pop(0)
        inputs = self.input_processor(
            text=task_descriptions,
            images={"images": primary_image},
            proprio_states=env_obs["states"],
            padding="max_length",
            max_length=self.max_prompt_length,
        )
        if all_images:
            all_wrist_inputs = [
                self.input_processor(
                    text=task_descriptions,
                    images={"images": wrist_image.unsqueeze(1)},
                    proprio_states=env_obs["states"],
                    padding="max_length",
                    max_length=self.max_prompt_length,
                )
                for wrist_image in all_images
            ]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"]]
                + [w["pixel_values"] for w in all_wrist_inputs],
                dim=1,
            )

        input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = inputs["attention_mask"].to(device=device, dtype=torch.bool)
        pixel_values = inputs["pixel_values"].to(device=device, dtype=precision)
        B, N, C, H, W = pixel_values.shape
        pixel_values = pixel_values.reshape(B, N * C, H, W)
        return input_ids, attention_mask, pixel_values

    # ---- BasePolicy interface -------------------------------------------------
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "vvla_openvla_oft is a rollout backend; keep actor.model.model_type="
            "openvla_oft for training / logprob recompute / value"
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        env_obs=None,
        calculate_logprobs=True,
        calculate_values=True,
        **kwargs,
    ):
        """native-OFT-contract rollout step: env obs -> (env actions, RL extras)."""
        do_sample = kwargs.pop("do_sample")
        temperature = float(kwargs["temperature"])
        top_k = int(kwargs["top_k"])

        if env_obs is not None:
            input_ids, attention_mask, pixel_values = self._encode_obs(env_obs)

        # sanity: BOS at 0, space token at the end (native asserts the same)
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(input_ids[:, -1] == _SPACE_TOKEN)

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        # ---- single-pass decode via the vvla engine --------------------------
        from vvla.policies.openvla_oft.processor_openvla_oft import OpenVLAOFTBatch

        B = input_ids.shape[0]
        batch = OpenVLAOFTBatch(
            input_ids=input_ids,
            attention_mask=attention_mask.to(torch.long),
            pixel_values=pixel_values,
            request_ids=[str(i) for i in range(B)],
        )
        # full forward (no prefill/decode split): bit-exact vs the native OFT forward, so
        # the PPO ratio at theta == theta_behavior is exactly 1. The split path
        # (encode_prefix + _decode) is retained on the policy as the CUDA-graph fast path
        # (lower bf16 precision — see docs / the split-vs-full diagnosis).
        logits, space_hidden = self.vvla_policy.action_logits_full(batch)
        idxs, logprob = self.vvla_policy.head.sample(
            logits, do_sample=do_sample, temperature=temperature, top_k=top_k
        )  # idxs [B, 56] absolute ids, logprob [B, 56]
        actions = self.vvla_policy.head.tokens_to_actions(idxs)  # [B, num_chunks, dim]

        chunk_action_tokens = idxs.reshape(B, self.num_action_chunks, self.action_dim)
        forward_inputs["action_tokens"] = chunk_action_tokens

        # ---- value (space-position hidden), or zeros when no value head ------
        if self.add_value_head and calculate_values:
            values = self.value_head(space_hidden)  # [B, output_dim]
        else:
            values = torch.zeros_like(logprob[..., :1])  # [B, 1]

        result = {
            "prev_logprobs": logprob,
            "prev_values": values,
            "forward_inputs": forward_inputs,
        }
        return actions, result


def get_model(cfg, torch_dtype=None):
    model = VvlaOpenVLAOFTForRLActionPrediction(cfg)
    return model.eval()
