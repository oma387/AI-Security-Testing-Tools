#!/usr/bin/env python3
"""Read-only inventory for Qdrant, ChromaDB, Milvus, and Weaviate.

Reports collection/class names, counts where available, and schema/vector
configuration summaries. It does NOT retrieve documents, payload values, or
embedding vectors. Use only against databases you own or are authorized to assess.

Optional dependencies are imported only for the selected backend.
Credentials are read from VECTOR_DB_TOKEN (or WEAVIATE_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a dict or an SDK model object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _vector_config_summary(config: Any) -> dict[str, Any]:
    """Return vector dimensions/config only; never returns stored vector data."""
    if config is None:
        return {}

    # Qdrant unnamed-vector configuration: {size, distance} or VectorParams.
    size = _value(config, "size")
    distance = _value(config, "distance")
    if size is not None:
        return {"default": {"dimensions": size, "distance": str(distance)}}

    # Qdrant named vectors: mapping vector name -> VectorParams.
    result: dict[str, Any] = {}
    if isinstance(config, dict):
        for name, params in config.items():
            result[str(name)] = {
                "dimensions": _value(params, "size"),
                "distance": str(_value(params, "distance", "unknown")),
            }
    else:
        result["configuration"] = str(type(config).__name__)
    return result


def inspect_qdrant(host: str, port: int, tls: bool, token: str | None) -> list[dict[str, Any]]:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("Install the Qdrant adapter: pip install 'qdrant-client>=1.9,<2'") from exc

    scheme = "https" if tls else "http"
    client = QdrantClient(url=f"{scheme}://{host}:{port}", api_key=token or None, timeout=10)
    results = []
    try:
        for item in client.get_collections().collections:
            name = item.name
            info = client.get_collection(name)
            count = client.count(collection_name=name, exact=False).count
            results.append({
                "name": name,
                "record_count_estimate": count,
                "status": str(info.status),
                "vectors": _vector_config_summary(info.config.params.vectors),
            })
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return results


def inspect_chroma(host: str, port: int, tls: bool, token: str | None) -> list[dict[str, Any]]:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Install the Chroma adapter: pip install 'chromadb>=0.5,<2'") from exc

    headers = {"Authorization": f"Bearer {token}"} if token else None
    client = chromadb.HttpClient(host=host, port=port, ssl=tls, headers=headers)
    results = []
    for item in client.list_collections():
        # Chroma SDK versions have returned either names or Collection objects.
        name = item if isinstance(item, str) else item.name
        collection = client.get_collection(name=name)
        count = collection.count()
        results.append({"name": name, "record_count": count})
    return results


def inspect_milvus(host: str, port: int, tls: bool, token: str | None) -> list[dict[str, Any]]:
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError("Install the Milvus adapter: pip install 'pymilvus>=2.4,<3'") from exc

    scheme = "https" if tls else "http"
    kwargs: dict[str, Any] = {"uri": f"{scheme}://{host}:{port}"}
    if token:
        kwargs["token"] = token
    client = MilvusClient(**kwargs)
    results = []
    try:
        for name in client.list_collections():
            description = client.describe_collection(collection_name=name)
            stats = client.get_collection_stats(collection_name=name)
            fields = []
            for field in description.get("fields", []):
                fields.append({
                    "name": field.get("name"),
                    "type": str(field.get("type", "unknown")),
                    "dimension": (field.get("params") or {}).get("dim"),
                })
            results.append({
                "name": name,
                "row_count": (stats or {}).get("row_count"),
                "fields": fields,
            })
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return results


def inspect_weaviate(host: str, port: int, tls: bool, token: str | None) -> list[dict[str, Any]]:
    """Read Weaviate's schema endpoint; no object/vector retrieval is performed."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Install requests: pip install requests") from exc

    scheme = "https" if tls else "http"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{scheme}://{host}:{port}/v1/schema"
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    schema = response.json()
    results = []
    for cls in schema.get("classes", []):
        results.append({
            "name": cls.get("class"),
            "vectorizer": cls.get("vectorizer"),
            "properties": [p.get("name") for p in cls.get("properties", [])],
            "note": "Schema inventory only; object counts are not queried.",
        })
    return results


def self_test() -> None:
    assert _vector_config_summary({"size": 1536, "distance": "Cosine"}) == {
        "default": {"dimensions": 1536, "distance": "Cosine"}
    }
    named = _vector_config_summary({"text": {"size": 384, "distance": "Dot"}})
    assert named["text"]["dimensions"] == 384
    assert named["text"]["distance"] == "Dot"
    assert _value({"name": "demo"}, "name") == "demo"
    assert _value(type("Obj", (), {"name": "demo"})(), "name") == "demo"
    print("SELF-TEST PASSED: config parsing and SDK-object helpers")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only vector database inventory (no record/vector export).")
    parser.add_argument("--db", choices=["qdrant", "chroma", "milvus", "weaviate"], required=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--tls", action="store_true", help="Use HTTPS/TLS where supported by the service/client.")
    parser.add_argument("--self-test", action="store_true", help="Run local tests without connecting to a database.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.db:
        parser.error("--db is required unless --self-test is used")

    default_ports = {"qdrant": 6333, "chroma": 8000, "milvus": 19530, "weaviate": 8080}
    port = args.port or default_ports[args.db]
    token = os.getenv("VECTOR_DB_TOKEN") or os.getenv("WEAVIATE_API_KEY")

    adapters = {
        "qdrant": inspect_qdrant,
        "chroma": inspect_chroma,
        "milvus": inspect_milvus,
        "weaviate": inspect_weaviate,
    }
    try:
        inventory = adapters[args.db](args.host, port, args.tls, token)
        print(json.dumps({"database": args.db, "host": args.host, "port": port, "collections": inventory}, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
