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

"""vvla rollout backend: wraps the vvla inference engine as an RLinf policy.

vvla (https://github.com/Longxmas/vvla) is an inference/rollout engine for
flow-matching VLA models. This adapter exposes a vvla-served pi0.5 as a
drop-in *rollout* policy, while the actor keeps training the reference
openpi implementation; the two are reconciled through weight sync.
"""

from .vvla_action_model import get_model

__all__ = ["get_model"]
