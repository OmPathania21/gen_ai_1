"""LLM client (Groq) with an on-disk response cache.

The cache makes experiments reproducible and keeps us inside free-tier rate
limits: identical (model, messages, temperature, sample index) requests are
answered from disk.
"""

import hashlib
import json
import time

from .config import settings


class LLM:
    def __init__(self, model: str | None = None, use_cache: bool = True):
        from groq import Groq

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Put it in the .env file in the project root.")
        self.client = Groq(api_key=settings.groq_api_key, max_retries=5)
        self.model = model or settings.llm_model
        self.use_cache = use_cache
        self.cache_dir = settings.cache_dir / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0          # real API calls made (cache misses)

    def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int | None = None,
             json_mode: bool = False, sample_idx: int = 0) -> str:
        """sample_idx distinguishes repeated stochastic samples of the same prompt in the cache."""
        key_src = json.dumps([self.model, messages, temperature, max_tokens, json_mode, sample_idx], sort_keys=True)
        path = self.cache_dir / f"{hashlib.sha256(key_src.encode()).hexdigest()}.json"
        if self.use_cache and path.exists():
            return json.loads(path.read_text())["content"]

        kwargs = dict(model=self.model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens or settings.max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model:
            kwargs["extra_body"] = {"reasoning_effort": settings.reasoning_effort}

        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:  # free tier is 8k tokens/min: wait for the window to reset
                if getattr(e, "status_code", None) != 429 or attempt == 5:
                    raise
                time.sleep(15 * (attempt + 1))

        self.calls += 1
        content = resp.choices[0].message.content or ""
        if self.use_cache:
            path.write_text(json.dumps({"content": content}))
        return content

    def chat_json(self, messages: list[dict], temperature: float = 0.0, **kw) -> dict:
        raw = self.chat(messages, temperature=temperature, json_mode=True, **kw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            return json.loads(raw[start:end + 1]) if start != -1 else {}
