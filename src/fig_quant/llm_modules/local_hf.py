"""Local in-process inference backend via Hugging Face transformers/PyTorch.

Used when ``USE_LOCAL=True``: runs entirely on local GPU/CPU/edge VRAM
with no outbound network call, for air-gapped or data-residency-sensitive
pod deployments (e.g. claim narratives that cannot leave the VPC).
"""

from __future__ import annotations

import dataclasses
import functools

from fig_quant.llm_modules.base import BaseLLM, LlmResponse

_DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"


@dataclasses.dataclass(slots=True)
class LocalHfLLM(BaseLLM):
    """Local HF Transformers text-generation backend.

    Attributes:
      model_name: Hugging Face Hub model identifier (e.g. Phi-3, Llama-3,
        Gemma family instruct checkpoints).
      device: Torch device string.
    """

    model_name: str = _DEFAULT_MODEL
    device: str = "cpu"

    @functools.cached_property
    def _pipeline(self):  # noqa: ANN202 - lazy import, external type
        import torch
        from transformers import pipeline

        return pipeline(
            "text-generation",
            model=self.model_name,
            device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> LlmResponse:
        """Generates a completion using the local HF pipeline.

        Args:
          prompt: Input prompt text.
          max_new_tokens: Maximum new tokens to generate.
          temperature: Sampling temperature.

        Returns:
          A normalized :class:`LlmResponse`.
        """
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            do_sample=temperature > 0,
            return_full_text=False,
        )
        text = outputs[0]["generated_text"]
        return LlmResponse(text=text, model_name=self.model_name)
