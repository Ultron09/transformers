# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Run exported generative models by orchestrating their ONNX graphs through `GenerationMixin.generate`.

`HfExporter.export_for_generation` produces one ONNX graph per component; these wrappers plug the graphs
back together and drive the generation loop. They are backend-light: you hand them already-created
`onnxruntime.InferenceSession`s and they discover everything else (which text input the decode graph
takes, how many attention masks, the KV-cache input/output pairs) from the graph signatures.

The goal is to run from artifacts alone — the exported graphs plus the saved config — without loading the
original checkpoint's weights. The only piece that still needs the model *type* is the modality precompute
(`OnnxModalityFeatures`), and it only introspects the module, so build that handle on the `meta` device
from the saved config (no weights allocated).

- `ExportedTextGenerator` — decoder-only text generation over a `decode` graph that takes `input_ids`.
- `ExportedMultimodalGenerator` — image / video / audio generation: a token-embedding graph + one
  features graph per modality are merged (`masked_scatter` at each modality's placeholder positions) into
  `inputs_embeds`, which the `decode` graph consumes. Modalities are described by `ModalityInput`, and
  `OnnxModalityFeatures` wraps a features graph + its precompute.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_leaves

from ..generation import GenerationMixin
from ..modeling_outputs import CausalLMOutputWithPast
from .utils import precompute_export_inputs


class _ExportedGenerationRuntime(GenerationMixin):
    """Shared decode-graph orchestration + the `GenerationMixin` plumbing a `PreTrainedModel` normally
    supplies. Subclasses decide what the decode graph's text input is (`input_ids` vs `inputs_embeds`)."""

    main_input_name = "input_ids"
    _supports_cache_class = True

    def __init__(self, config, generation_config, decode_session, device="cpu", dtype=torch.float32):
        self.config = config
        self.generation_config = generation_config
        self._device = torch.device(device)
        self._dtype = dtype
        self._decode = decode_session
        self._decode_outputs = [o.name for o in decode_session.get_outputs()]
        input_names = [i.name for i in decode_session.get_inputs()]
        # The exporter names the decode graph's inputs after the forward kwargs it captured, so the graph
        # itself tells us how to feed it: which text input, how many masks, and the KV-cache pairs.
        self._text_input = "inputs_embeds" if "inputs_embeds" in input_names else "input_ids"
        self._mask_inputs = [n for n in input_names if n == "attention_mask" or n.startswith("attention_mask.")]
        self._cache_names = [n[len("input.") :] for n in input_names if n.startswith("input.")]

    # ── GenerationMixin plumbing (a real PreTrainedModel provides all of this) ──
    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def can_generate(self):
        return True

    def is_remote_code(self):
        return False

    def get_experts_implementation(self):
        return {}

    def get_output_embeddings(self):
        return None

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    # ── decode orchestration ──
    def _decoder_text_input(self, input_ids, **kwargs):
        """Return the tensor for the decode graph's text input. Base case (text): the raw `input_ids`.
        The image-text subclass overrides this to return vision-merged `inputs_embeds`."""
        return input_ids

    def _attention_mask_feed(self, attention_mask, position_ids, cache_len):
        """Feed the decode graph's mask input(s). `generate` hands us either a dict of 4D bool masks (one
        per attention type, for mixed full/sliding models) or a single 4D mask; when it drops the mask as
        redundant (unpadded sequence on a single attention type) we rebuild a full causal mask from the
        positions. Assumes no left-padding — the common single-sequence case."""
        if attention_mask is None:
            positions = position_ids[0]
            mask = (torch.arange(cache_len, device=positions.device) <= positions[:, None])[None, None]
            return {self._mask_inputs[0]: mask.numpy()}
        if isinstance(attention_mask, dict):
            return {f"attention_mask.{layer_type}": mask.numpy() for layer_type, mask in attention_mask.items()}
        return {self._mask_inputs[0]: attention_mask.numpy()}

    def forward(self, input_ids, past_key_values, position_ids, attention_mask=None, **kwargs):
        text = self._decoder_text_input(input_ids, **kwargs)
        feed = {self._text_input: text.numpy(), "position_ids": position_ids.numpy()}
        feed.update(self._attention_mask_feed(attention_mask, position_ids, past_key_values.get_max_length()))

        # KV buffers travel as matched `input.<name>` / `output.<name>` pairs, in pytree-leaf order.
        cache_leaves = [t for t in tree_leaves(past_key_values) if isinstance(t, torch.Tensor)]
        feed.update({f"input.{name}": t.detach().numpy() for name, t in zip(self._cache_names, cache_leaves)})

        outputs = dict(zip(self._decode_outputs, self._decode.run(None, feed)))

        # Write the mutated cache back so the next step continues from it.
        for name, tensor in zip(self._cache_names, cache_leaves):
            tensor.copy_(torch.from_numpy(outputs[f"output.{name}"]))
        # Sliding layers track length in a plain python int (not a pytree tensor, so the write-back above
        # misses it); sync it from the tensor counter or the next step's mask/position bookkeeping goes stale.
        for layer in past_key_values.layers:
            if hasattr(layer, "cumulative_length_int"):
                layer.cumulative_length_int = int(layer.cumulative_length)

        return CausalLMOutputWithPast(logits=torch.from_numpy(outputs["logits"]), past_key_values=past_key_values)


class ExportedTextGenerator(_ExportedGenerationRuntime):
    """Decoder-only text generation over an exported `decode` graph that takes `input_ids`.

    Example:
        programs = OnnxExporter().export_for_generation(model, inputs, OnnxConfig(dynamic=True),
                                                        generation_config=static_cache_config,
                                                        multi_token_decode=True)
        decode_session = ort.InferenceSession(programs["decode"].model_proto.SerializeToString())
        runner = ExportedTextGenerator(model.config, model.generation_config, decode_session)
        ids = runner.generate(input_ids=prompt, past_key_values=static_cache, max_new_tokens=32)
    """


@dataclass
class ModalityInput:
    """One input modality for `ExportedMultimodalGenerator`:

    - `token_id`: the placeholder id in `input_ids` its features scatter into (`config.image_token_id`, …),
    - `features`: a callable turning this modality's generate kwargs into `[num_tokens, hidden]` embeds
      (see `OnnxModalityFeatures`),
    - `input_keys`: the generate kwargs that belong to it (e.g. `("pixel_values", "image_grid_thw")`,
      `("input_features", "feature_attention_mask")`).
    """

    token_id: int
    features: Callable
    input_keys: tuple


class ExportedMultimodalGenerator(_ExportedGenerationRuntime):
    """Multi-modal generation over any mix of image / video / audio. On the prefill step it embeds
    `input_ids` (via `embed_session`), computes each modality's features, and scatters them into their
    placeholder rows before the decode graph runs; decode steps are text-only. The decode graph takes
    `inputs_embeds`.

    Args:
        embed_session: ONNX session mapping `input_ids -> inputs_embeds` (placeholder ids handled inside).
        modalities: list of `ModalityInput`, one per input modality the model consumes.
    """

    def __init__(
        self,
        config,
        generation_config,
        decode_session,
        embed_session,
        modalities,
        *,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__(config, generation_config, decode_session, device=device, dtype=dtype)
        self._embed = embed_session
        self._modalities = list(modalities)
        self._modality_keys = {key for modality in self._modalities for key in modality.input_keys}

    def _validate_model_kwargs(self, model_kwargs):
        # The modality inputs (pixel_values, input_features, …) are consumed in `_decoder_text_input`, not
        # named on `forward`; drop them before `generate`'s "unused kwargs" check.
        super()._validate_model_kwargs({k: v for k, v in model_kwargs.items() if k not in self._modality_keys})

    def _decoder_text_input(self, input_ids, **kwargs):
        inputs_embeds = torch.from_numpy(self._embed.run(None, {"input_ids": input_ids.numpy()})[0])
        for modality in self._modalities:
            # Presence keys on the primary input (`input_keys[0]`: pixel_values / input_features); `generate`
            # drops it after prefill but may keep a stale grid kwarg, so don't trigger on the aux keys.
            if kwargs.get(modality.input_keys[0]) is None:
                continue
            feed = {key: kwargs[key] for key in modality.input_keys if kwargs.get(key) is not None}
            features = torch.from_numpy(modality.features(feed))
            mask = (input_ids == modality.token_id).unsqueeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.dtype))
        return inputs_embeds


class OnnxModalityFeatures:
    """Runs an exported `get_<modality>_features` graph to produce modality embeds, injecting the
    grid-derived precompute (`cu_seqlens` / `window_index` / `pixel_shuffle_index` / audio chunking / …)
    the graph expects. Maps generate's grid kwarg (`image_grid_thw` / `video_grid_thw`) to the graph's
    `grid_thw` input, and feeds only the kwargs the graph actually declares.

    `precompute_model` is only introspected (never run — the modality inputs carry no `input_ids`, so the
    outer-LLM rope branch stays off); build it on the `meta` device from the saved config to keep runtime
    free of the original checkpoint's weights.
    """

    def __init__(self, features_session, precompute_model, *, grid_kwarg=None):
        self._session = features_session
        self._precompute_model = precompute_model
        self._grid_kwarg = grid_kwarg
        self._input_names = {i.name for i in features_session.get_inputs()}

    def __call__(self, modality_kwargs):
        inputs = dict(modality_kwargs)
        if self._grid_kwarg is not None and self._grid_kwarg in inputs:
            inputs["grid_thw"] = inputs.pop(self._grid_kwarg)
        precompute_export_inputs(self._precompute_model, inputs)
        feed = {k: v.numpy() for k, v in inputs.items() if k in self._input_names and isinstance(v, torch.Tensor)}
        return self._session.run(None, feed)[0]
