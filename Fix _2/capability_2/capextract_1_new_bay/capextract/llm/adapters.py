"""
Pluggable LLM adapters.

All adapters expose two callables:
  - llm_call(system_prompt, user_message) -> str
      For intent analysis (single-turn, structured output)
  - stream_code(prompt, on_chunk, on_done)
      For streaming code generation token by token

Usage:
    from capextract.llm.adapters import get_adapter
    adapter = get_adapter("claude")          # or "openai", "ollama"
    fn = adapter.llm_call
    gen = adapter.stream_code
"""

from __future__ import annotations
import os
import json
from abc import ABC, abstractmethod
from typing import Callable


class LLMAdapter(ABC):
    @abstractmethod
    def llm_call(self, system_prompt: str, user_message: str) -> str:
        """Single-turn call — returns complete response string."""
        ...

    @abstractmethod
    def stream_code(
        self,
        prompt: str,
        on_chunk: Callable[[str], None],
        on_done: Callable[[str], None],
    ):
        """
        Stream code generation token by token.
        on_chunk(token_text) called for each token/chunk.
        on_done(full_code) called once when generation finishes.
        """
        ...


# ─────────────────────────────────────────────────────────────────
# Claude adapter
# ─────────────────────────────────────────────────────────────────

class ClaudeAdapter(LLMAdapter):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def llm_call(self, system_prompt: str, user_message: str) -> str:
        import anthropic
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return msg.content[0].text

    def stream_code(self, prompt: str, on_chunk, on_done):
        full = []
        with self._client.messages.stream(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                full.append(text)
                on_chunk(text)
        on_done("".join(full))


# ─────────────────────────────────────────────────────────────────
# OpenAI adapter
# ─────────────────────────────────────────────────────────────────

class OpenAIAdapter(LLMAdapter):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))  # type: ignore
        self.model = model

    def llm_call(self, system_prompt: str, user_message: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    def stream_code(self, prompt: str, on_chunk, on_done):
        full = []
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full.append(delta)
                on_chunk(delta)
        on_done("".join(full))


# ─────────────────────────────────────────────────────────────────
# Groq adapter
# ─────────────────────────────────────────────────────────────────

class GroqAdapter(LLMAdapter):
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("pip install groq")
        self._client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model

    def llm_call(self, system_prompt: str, user_message: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    def stream_code(self, prompt: str, on_chunk, on_done):
        full = []
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full.append(delta)
                on_chunk(delta)
        on_done("".join(full))


# ─────────────────────────────────────────────────────────────────
# Ollama adapter  (local models)
# ─────────────────────────────────────────────────────────────────

class OllamaAdapter(LLMAdapter):
    def __init__(self, model: str = "codellama", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def llm_call(self, system_prompt: str, user_message: str) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "prompt": f"{system_prompt}\n\nUser: {user_message}",
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["response"]

    def stream_code(self, prompt: str, on_chunk, on_done):
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        full = []
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                if line.strip():
                    obj = json.loads(line)
                    tok = obj.get("response", "")
                    if tok:
                        full.append(tok)
                        on_chunk(tok)
        on_done("".join(full))


# ─────────────────────────────────────────────────────────────────
# Mock adapter for testing (no API key needed)
# ─────────────────────────────────────────────────────────────────

class MockAdapter(LLMAdapter):
    """Returns deterministic responses for testing without any API."""

    MOCK_INTENT = json.dumps({
        "goal": "Generate code as requested",
        "expected_capabilities": [],
        "scope_constraints": [],
        "resource_hints": [],
        "ambiguities": [],
        "detected_language": "python",
    })

    def __init__(self, mock_code: str = ""):
        self.mock_code = mock_code or "print('hello world')"

    def llm_call(self, system_prompt: str, user_message: str) -> str:
        # Try to extract a sensible intent from the prompt
        prompt_lower = user_message.lower()
        caps = []
        constraints = []
        hints = []
        lang = "python"

        if any(k in prompt_lower for k in ["csv","dataframe","pandas","analyze","plot"]):
            caps.append("DataAnalytics")
            hints.append("filesystem")
        if any(k in prompt_lower for k in ["train","model","sklearn","torch","tensorflow"]):
            caps.append("MachineLearning")
            hints.append("gpu")
        if any(k in prompt_lower for k in ["scrape","crawl","requests","beautifulsoup"]):
            caps.append("WebScraping")
            hints.append("network")
        if any(k in prompt_lower for k in ["database","sql","sqlite","postgres"]):
            caps.append("DatabaseOperations")
            hints.append("database")
        if "read" in prompt_lower and "network" not in prompt_lower:
            constraints.append("local_only")
        if "javascript" in prompt_lower or "node" in prompt_lower:
            lang = "javascript"
        elif "java" in prompt_lower:
            lang = "java"

        return json.dumps({
            "goal": user_message[:120],
            "expected_capabilities": caps,
            "scope_constraints": constraints,
            "resource_hints": hints or ["filesystem"],
            "ambiguities": [],
            "detected_language": lang,
        })

    def stream_code(self, prompt: str, on_chunk, on_done):
        # Stream the mock code word by word
        import time
        words = self.mock_code.split(" ")
        full = []
        for word in words:
            chunk = word + " "
            full.append(chunk)
            on_chunk(chunk)
            time.sleep(0.01)
        on_done("".join(full))


# ─────────────────────────────────────────────────────────────────
# DeepSeek Adapter for local DeepSeek-Coder-V2-Lite-Instruct
# ─────────────────────────────────────────────────────────────────

class DeepSeekAdapter(LLMAdapter):
    """Loads and streams DeepSeek-Coder-V2-Lite-Instruct locally using Transformers."""

    def __init__(self, model_path: str = r"C:\Users\L509\Desktop\llm project\DeepSeek-Coder-V2-Lite-Instruct"):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("pip install torch transformers")

        self.model_path = model_path
        print(f"[DeepSeekAdapter] Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"[DeepSeekAdapter] Loading model from {model_path}...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
        print("[DeepSeekAdapter] Model loaded successfully.")

    def llm_call(self, system_prompt: str, user_message: str) -> str:
        import torch
        prompt = f"{system_prompt}\n\nUser Question:\n{user_message}\n\nAssistant:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def stream_code(self, prompt: str, on_chunk, on_done):
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = []
        for new_text in streamer:
            full_response.append(new_text)
            on_chunk(new_text)

        thread.join()
        on_done("".join(full_response))


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────

def get_adapter(name: str, **kwargs) -> LLMAdapter:
    """
    name: "claude" | "openai" | "groq" | "ollama" | "mock" | "deepseek"
    kwargs: passed to the adapter constructor
    """
    adapters = {
        "claude":   ClaudeAdapter,
        "openai":   OpenAIAdapter,
        "groq":     GroqAdapter,
        "ollama":   OllamaAdapter,
        "mock":     MockAdapter,
        "deepseek": DeepSeekAdapter,
    }
    cls = adapters.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown adapter '{name}'. Choose from: {list(adapters)}")
    return cls(**kwargs)

