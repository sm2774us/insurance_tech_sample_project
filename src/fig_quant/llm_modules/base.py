"""Unified LLM interface blueprint: swappable local vs. external routing.

Application code depends only on :class:`BaseLLM`; concrete backends
(local HF transformers vs. external cloud router) are selected at runtime
via ``USE_LOCAL``, with zero call-site changes.
"""

from __future__ import annotations

import abc
import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class LlmResponse:
    """A normalized LLM completion response.

    Attributes:
      text: Generated completion text.
      model_name: Identifier of the backend model that served the request.
      prompt_tokens: Prompt token count, if the backend reports it.
      completion_tokens: Completion token count, if the backend reports it.
    """

    text: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BaseLLM(abc.ABC):
    """Abstract blueprint every LLM backend implements."""

    @abc.abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> LlmResponse:
        """Generates a completion for ``prompt``.

        Args:
          prompt: Input prompt text.
          max_new_tokens: Maximum tokens to generate.
          temperature: Sampling temperature.

        Returns:
          A normalized :class:`LlmResponse`.
        """
        raise NotImplementedError


def get_llm(use_local: bool, **kwargs: object) -> BaseLLM:
    """Factory selecting the local or external LLM backend.

    Args:
      use_local: If ``True``, routes to in-process HF transformers
        inference; if ``False``, routes to the external LiteLLM cloud
        proxy.
      **kwargs: Backend-specific constructor arguments.

    Returns:
      A concrete :class:`BaseLLM` implementation.
    """
    if use_local:
        from fig_quant.llm_modules.local_hf import LocalHfLLM

        return LocalHfLLM(**kwargs)
    from fig_quant.llm_modules.external_lite import ExternalLiteLLM

    return ExternalLiteLLM(**kwargs)
