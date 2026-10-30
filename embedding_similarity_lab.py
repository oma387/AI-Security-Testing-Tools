#!/usr/bin/env python3
"""Compare synthetic candidate text against an embedding using a chosen model.

Providers: OpenAI, Sentence-Transformers (Hugging Face), and Ollama.
This is candidate similarity analysis, NOT universal embedding inversion. It
cannot reliably recover arbitrary original text from an embedding. Use only
vectors and test material you own or are explicitly authorized to assess.

Credentials:
  OpenAI: OPENAI_API_KEY
  Ollama: local Ollama server (no API key by default)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: target={len(a)}, candidate={len(b)}")
    if not a:
        raise ValueError("Vectors must not be empty")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) ** 2 for x in a))
    norm_b = math.sqrt(sum(float(y) ** 2 for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def validate_vector(vector: object, label: str) -> list[float]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError(f"{label} must be a non-empty JSON array of numbers")
    result: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}[{index}] is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] is not finite")
        result.append(number)
    return result


class Embedder:
    def __init__(self, provider: str, model: str, ollama_url: str):
        self.provider = provider
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self._model_obj = None
        self._openai_client = None

    def embed(self, text: str) -> list[float]:
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install OpenAI support: pip install openai") from exc
            if self._openai_client is None:
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY is not set")
                self._openai_client = OpenAI()
            response = self._openai_client.embeddings.create(model=self.model, input=text)
            return validate_vector(response.data[0].embedding, "OpenAI embedding")

        if self.provider == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install Hugging Face support: pip install sentence-transformers") from exc
            if self._model_obj is None:
                self._model_obj = SentenceTransformer(self.model)
            vector = self._model_obj.encode(text, normalize_embeddings=False)
            return validate_vector(vector.tolist(), "Hugging Face embedding")

        if self.provider == "ollama":
            try:
                import requests
            except ImportError as exc:
                raise RuntimeError("Install Ollama HTTP support: pip install requests") from exc
            # Current Ollama endpoint. Older installations may only support /api/embeddings.
            response = requests.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=60,
            )
            if response.status_code == 404:
                response = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=60,
                )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload.get("embeddings"), list):
                vector = payload["embeddings"][0]
            else:
                vector = payload.get("embedding")
            return validate_vector(vector, "Ollama embedding")

        raise ValueError(f"Unsupported provider: {self.provider}")


def load_target_vector(path: str) -> list[float]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    # Accept either [0.1, ...] or {"embedding": [0.1, ...]}.
    if isinstance(payload, dict):
        if "model" in payload:
            print(f"Target file model metadata: {payload['model']}", file=sys.stderr)
        payload = payload.get("embedding")
    return validate_vector(payload, "Target vector")


def load_candidates(path: str | None) -> list[str]:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            candidates = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    else:
        candidates = [
            "Synthetic IT password reset policy for the Northstar test environment.",
            "Synthetic employee travel reimbursement policy for the Northstar test environment.",
            "Synthetic helpdesk account recovery procedure for the Northstar test environment.",
        ]
    if not candidates:
        raise ValueError("No candidate phrases were supplied")
    return candidates


def self_test() -> None:
    assert abs(cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9
    try:
        cosine_similarity([1, 2], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("Dimension mismatch was not rejected")
    try:
        validate_vector([1, float("nan")], "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite value was not rejected")
    print("SELF-TEST PASSED: cosine similarity, dimension checks, and vector validation")


def main() -> int:
    parser = argparse.ArgumentParser(description="Model-specific embedding similarity experiment; not text inversion.")
    parser.add_argument("--provider", choices=["openai", "huggingface", "ollama"], required=False)
    parser.add_argument("--model")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target-text", help="Synthetic/authorized text to embed as the reference.")
    group.add_argument("--target-vector-file", help="JSON file containing a vector generated by the same model.")
    parser.add_argument("--candidates-file", help="Text file with one candidate phrase per line.")
    parser.add_argument("--output", default="embedding_similarity_results.csv")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.provider or not args.model:
        parser.error("--provider and --model are required unless --self-test is used")
    if not args.target_text and not args.target_vector_file:
        parser.error("Provide --target-text or --target-vector-file")

    try:
        embedder = Embedder(args.provider, args.model, args.ollama_url)
        if args.target_vector_file:
            target = load_target_vector(args.target_vector_file)
            print("WARNING: target-vector compatibility depends on using the exact same embedding model and settings.")
        else:
            target = embedder.embed(args.target_text)

        candidates = load_candidates(args.candidates_file)
        rows = []
        for candidate in candidates:
            candidate_vector = embedder.embed(candidate)
            score = cosine_similarity(target, candidate_vector)
            rows.append((score, candidate))

        rows.sort(key=lambda item: item[0], reverse=True)
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cosine_similarity", "candidate"])
            writer.writerows(rows)

        print(f"Provider: {args.provider}; model: {args.model}; dimensions: {len(target)}")
        for score, candidate in rows:
            print(f"{score:.6f}  {candidate}")
        print(f"Saved results to {args.output}")
        print("Reminder: similarity ranking is not recovery of the original text.")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
