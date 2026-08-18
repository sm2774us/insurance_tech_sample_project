"""External cloud LLM router via the LiteLLM standard proxy.

Used when ``USE_LOCAL=False``: normalizes API requests across hosted
providers (OpenAI, Anthropic, etc.) behind one async I/O interface, so
application code never depends on a specific vendor SDK.
"""

from __future__ import annotations

import dataclasses

from fig_quant.llm_modules.base import BaseLLM, LlmResponse

_DEFAULT_MODEL = "claude-sonnet-5"


@dataclasses.dataclass(slots=True)
class ExternalLiteLLM(BaseLLM):
    """Hosted-provider backend routed through LiteLLM.

    Attributes:
      model_name: LiteLLM-normalized model identifier
        (e.g. ``"claude-sonnet-5"``, ``"gpt-4.1"``).
      timeout_sec: Request timeout in seconds.
    """

    model_name: str = _DEFAULT_MODEL
    timeout_sec: float = 30.0

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2) -> LlmResponse:
        """Generates a completion via the LiteLLM standard proxy.

        Args:
          prompt: Input prompt text.
          max_new_tokens: Maximum new tokens to generate.
          temperature: Sampling temperature.

        Returns:
          A normalized :class:`LlmResponse`.

        Raises:
          RuntimeError: If the provider request fails.
        """
        import litellm

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                timeout=self.timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise RuntimeError(f"External LLM request failed: {exc}") from exc

        choice = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        return LlmResponse(
            text=choice,
            model_name=self.model_name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
