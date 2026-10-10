"""Test doubles: a deterministic fake embedder and a fake LLM, so the tests run offline
without downloading models or needing Ollama."""
import hashlib
import re

import numpy as np


class FakeEmbedder:
    """Bag-of-words hashing embedder (64 dims): texts sharing words get similar vectors."""
    dim = 64

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False, **_):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return out


class FakeLLM:
    """Records calls; returns canned replies depending on which agent is calling."""

    def __init__(self):
        self.calls = []
        self.prompts = []  # (system prompt, user prompt) of every call, to check what the model was given

    def generate(self, prompt, system_instruction=None, temperature=0.3, max_retries=3):
        self.calls.append(("generate", (system_instruction or "")[:40]))
        self.prompts.append((system_instruction or "", prompt))
        s = system_instruction or ""
        if "Query Optimizer" in s:
            return "optimized: " + prompt.split("Original query:")[-1].split("\n")[0].strip()
        if "Router Agent" in s:
            return "BOTH"
        if "Report Analysis Agent" in s:
            return '{"HbA1c": {"value": 7.8, "unit": "%"}, "FBG": {"value": 142, "unit": "mg/dL"}}'
        if "Verification Agent" in s and "Return ONLY a JSON" in s:
            return '{"verified": true, "issues": []}'
        return "Answer text [Source: a.txt]. This is decision support only."

    def generate_stream(self, prompt, system_instruction=None, temperature=0.3):
        self.calls.append(("stream", (system_instruction or "")[:40]))
        self.prompts.append((system_instruction or "", prompt))
        for piece in ["Streamed ", "answer ", "[Source: a.txt]."]:
            yield piece
