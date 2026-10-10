"""Naming a sub-topic: keywords in, a readable root-cause label out.

A cluster is useless to a service-desk manager until it has a name, and the
gap between a good name and a bad one is large. ``poste / ecran / bleu`` is a
term list; *"Ecran bleu recurrent apres la mise a jour Windows"* is something a
problem manager can act on.

Three strategies, in increasing order of cost:

``TemplateLabeller`` (default, offline)
    Use the shortest representative ticket title as the name. It costs nothing,
    needs no model, and is surprisingly hard to beat: a real title written by a
    user already *is* a description of the symptom.

``LlmLabeller.local(...)``
    A small instruct model through ``transformers``. Generalises the examples
    into one phrase instead of copying one of them.

``LlmLabeller.openai_compatible(...)``
    Any OpenAI-compatible endpoint - vLLM, Ollama, Azure OpenAI, an internal
    gateway. Chosen over a vendor SDK because in a constrained environment the
    only reachable model is usually one somebody hosts internally.

All three satisfy the same one-method interface, so the discovery code does not
know which is in use, and a deployment that cannot reach any model still gets
names.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from itsm.logging import get_logger

logger = get_logger("analysis.topic_labeller")

PROMPT = """You are naming a cluster of IT service-desk tickets so a problem \
manager can recognise the underlying issue.

Ticket category: {category}
Frequent terms: {keywords}
Example ticket titles:
{examples}

Write ONE short noun phrase (3 to 8 words) naming the underlying problem these \
tickets share. Write it in the same language as the example titles. Do not \
add quotes, punctuation or any explanation. Answer with the phrase only."""


class Labeller(Protocol):
    """Anything that can name a sub-topic."""

    def label(self, keywords: list[str], examples: list[str], category: str) -> str:
        """Return a short human-readable name for the sub-topic."""
        ...


def _clean(text: str, fallback: str) -> str:
    """Strip the decoration models add around a short answer."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    first = re.sub(r'^["\'`*\s-]+|["\'`*\s.]+$', "", first)
    first = re.sub(r"^(label|nom|name|phrase)\s*[:\-]\s*", "", first, flags=re.IGNORECASE)
    # A model that ignores the instruction and writes a paragraph is worse than
    # the offline fallback, so length is a rejection criterion.
    return first[:90] if 3 <= len(first) <= 90 else fallback


class TemplateLabeller:
    """Offline labeller: the shortest representative ticket title."""

    name = "template"

    def label(self, keywords: list[str], examples: list[str], category: str) -> str:
        """Return the shortest example, or the top keywords as a last resort."""
        if examples:
            return min(examples, key=len)[:90].strip()
        return " / ".join(keywords[:3]) or "unlabelled"


class LlmLabeller:
    """Labeller backed by any text-generating callable.

    Args:
        generate: Maps a prompt to a completion.
        name: Identifier recorded in the report, for provenance.
    """

    def __init__(self, generate: Callable[[str], str], name: str = "llm") -> None:
        self.generate = generate
        self.name = name
        self._fallback = TemplateLabeller()

    def label(self, keywords: list[str], examples: list[str], category: str) -> str:
        """Generate a name, falling back to the offline labeller on any failure.

        A labelling step must never be able to fail a pipeline: an unnamed
        cluster is a cosmetic problem, a crashed analysis is not.
        """
        fallback = self._fallback.label(keywords, examples, category)
        prompt = PROMPT.format(
            category=category or "unknown",
            keywords=", ".join(keywords[:8]) or "(none)",
            examples="\n".join(f"- {example}" for example in examples[:5]),
        )
        try:
            return _clean(self.generate(prompt), fallback)
        except Exception as error:  # never fail the pipeline on a cosmetic step
            logger.warning("llm labelling failed (%s); using the offline label", error)
            return fallback

    @classmethod
    def local(cls, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", **kwargs: Any) -> LlmLabeller:
        """Build a labeller around a local ``transformers`` instruct model."""
        from transformers import pipeline

        generator = pipeline("text-generation", model=model_name, **kwargs)

        def generate(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}]
            output = generator(messages, max_new_tokens=32, do_sample=False)
            content = output[0]["generated_text"]
            return content[-1]["content"] if isinstance(content, list) else str(content)

        return cls(generate, name=f"local:{model_name}")

    @classmethod
    def openai_compatible(
        cls, base_url: str, model: str, api_key: str = "", timeout: int = 30
    ) -> LlmLabeller:
        """Build a labeller against an OpenAI-compatible chat endpoint."""
        import requests

        def generate(prompt: str) -> str:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 32,
                    "temperature": 0.0,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"])

        return cls(generate, name=f"openai:{model}")


def build_labeller(kind: str = "template", **kwargs: Any) -> Labeller:
    """Return the labeller matching ``kind`` (``template``, ``local``, ``api``)."""
    if kind == "local":
        return LlmLabeller.local(**kwargs)
    if kind == "api":
        return LlmLabeller.openai_compatible(**kwargs)
    return TemplateLabeller()
