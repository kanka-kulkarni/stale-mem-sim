"""Minimal client for a local Ollama model, with an on-disk response cache.

Used only by optional LLM components (the llm impact table and the per-session llm policy).
The core simulator never imports this module.

Calls use temperature 0 and a fixed seed, ask for JSON output, and are cached by a
hash of (model, prompt, options), so reruns are free and deterministic. Every call
reports latency and token counts.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_URL = "http://localhost:11434"
Transport = Callable[[str, dict, float], dict]


def _requests_transport(url: str, payload: dict, timeout: float) -> dict:
    import requests  # optional dependency, only needed for real calls

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


@dataclass(frozen=True)
class LLMResponse:
    """One model reply plus call metadata."""

    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cached: bool


class OllamaClient:
    """Calls Ollama's /api/generate; caches replies in a JSON-lines file."""

    def __init__(self, model: str, cache_path: Path | None = None, url: str = DEFAULT_URL,
                 temperature: float = 0.0, seed: int = 0, timeout: float = 600.0,
                 transport: Transport | None = None, retries: int = 3,
                 retry_wait_s: float = 30.0, think: bool | None = None) -> None:
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self.retries, self.retry_wait_s = retries, retry_wait_s
        self.options = {"temperature": temperature, "seed": seed}
        # None = Ollama's default; only an explicit value enters payloads and cache keys
        self.think = think
        self.transport = transport or _requests_transport
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path is not None and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec

    def _call_with_retries(self, payload: dict) -> dict:
        """Call the transport, retrying transient failures (for unattended runs).
        Latency includes any retry waits; failed attempts are not cached."""
        for attempt in range(self.retries + 1):
            try:
                return self.transport(f"{self.url}/api/generate", payload, self.timeout)
            except Exception:  # noqa: BLE001 - network/server errors of any kind
                if attempt == self.retries:
                    raise
                time.sleep(self.retry_wait_s * (attempt + 1))
        raise AssertionError("unreachable")

    def key(self, prompt: str) -> str:
        """Cache key: hash of model, prompt and sampling options."""
        spec = {"model": self.model, "prompt": prompt, "options": self.options,
                "format": "json"}
        if self.think is not None:
            spec["think"] = self.think
        blob = json.dumps(spec, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def generate(self, prompt: str) -> LLMResponse:
        """Return the model's reply, from cache if available."""
        k = self.key(prompt)
        if k in self.cache:
            r = self.cache[k]
            return LLMResponse(r["text"], r["latency_s"], r["prompt_tokens"],
                               r["completion_tokens"], cached=True)
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "format": "json", "options": self.options}
        if self.think is not None:
            payload["think"] = self.think
        start = time.perf_counter()
        out = self._call_with_retries(payload)
        latency = time.perf_counter() - start
        rec = {"key": k, "model": self.model, "prompt": prompt,
               "text": out.get("response", ""), "latency_s": latency,
               "prompt_tokens": int(out.get("prompt_eval_count", 0)),
               "completion_tokens": int(out.get("eval_count", 0))}
        self.cache[k] = rec
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return LLMResponse(rec["text"], latency, rec["prompt_tokens"],
                           rec["completion_tokens"], cached=False)
