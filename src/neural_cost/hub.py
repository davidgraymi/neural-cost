"""Zero-code Hugging Face model configuration ingestion and architecture resolver."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class HFModelConfig:
    """Standardized neural architecture dimensions extracted from a Hugging Face model config."""

    model_id: str
    model_type: str
    architecture: str  # 'transformer', 'moe', or 'ssm'
    embed_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_dim: int
    vocab_size: int
    max_position_embeddings: int | None
    dtype: str
    num_experts: int | None = None
    top_k: int | None = None
    state_dim: int | None = None
    expand_factor: int | None = None
    conv_kernel_size: int | None = None
    raw_config: dict[str, Any] | None = None


def get_default_cache_dir() -> Path:
    """Return default cache directory for downloaded Hugging Face model configs."""
    cache_base = os.environ.get("NEURAL_COST_CACHE_DIR")
    if cache_base:
        return Path(cache_base)
    home = Path.home()
    return home / ".cache" / "neural-cost" / "hub"


def fetch_hf_config(
    model_id_or_path: str,
    *,
    revision: str = "main",
    token: str | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch model configuration JSON from local path, cache, or remote Hugging Face Hub.

    Parameters:
        model_id_or_path: Hugging Face model ID (e.g. 'meta-llama/Meta-Llama-3-8B', 'hf:mistralai/Mistral-7B-v0.1')
                          or a path to a local directory or config.json file.
        revision: Git branch, tag, or commit hash on Hugging Face Hub (default: 'main').
        token: Optional Hugging Face access token (uses HF_TOKEN environment variable if omitted).
        cache_dir: Optional custom local cache directory.
    """
    cleaned = model_id_or_path.strip()
    if cleaned.lower().startswith("hf:"):
        cleaned = cleaned[3:].strip()

    # 1. Local file or directory check
    local_candidate = Path(cleaned).expanduser()
    if local_candidate.is_file():
        with open(local_candidate, encoding="utf-8") as f:
            return json.load(f)
    if local_candidate.is_dir():
        cfg_file = local_candidate / "config.json"
        if cfg_file.is_file():
            with open(cfg_file, encoding="utf-8") as f:
                return json.load(f)

    # 2. Local cache check
    c_dir = cache_dir if cache_dir is not None else get_default_cache_dir()
    safe_model_dir = cleaned.replace("/", "--")
    cached_file = c_dir / safe_model_dir / f"{revision}_config.json"
    if cached_file.is_file():
        try:
            with open(cached_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # 3. Remote fetch from Hugging Face Hub
    url = f"https://huggingface.co/{cleaned}/raw/{revision}/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "neural-cost"})
    auth_token = token or os.environ.get("HF_TOKEN")
    if auth_token:
        req.add_header("Authorization", f"Bearer {auth_token}")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
            config = json.loads(data)
    except urllib.error.HTTPError as exc:
        if exc.code == 401 or exc.code == 403:
            raise PermissionError(
                f"Access denied to '{cleaned}' (HTTP {exc.code}). "
                "This repository may be private or gated (e.g. LLaMA). "
                "Provide a token via HF_TOKEN environment variable or --token."
            ) from exc
        if exc.code == 404:
            raise FileNotFoundError(
                f"Model or config not found on Hugging Face Hub: '{cleaned}' (HTTP 404 at {url})."
            ) from exc
        raise RuntimeError(
            f"HTTP error {exc.code} fetching config from {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Network error connecting to Hugging Face Hub for '{cleaned}': {exc.reason}"
        ) from exc

    # Cache successful fetch
    try:
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cached_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except OSError:
        pass

    return config


def parse_hf_config(config_dict: dict[str, Any], model_id: str = "custom") -> HFModelConfig:
    """Parse raw Hugging Face config dictionary into normalized architectural specifications."""
    model_type = str(config_dict.get("model_type", "transformer")).lower()

    # Normalize architecture category
    if (
        model_type
        in (
            "mixtral",
            "deepseek_v2",
            "deepseek_v3",
            "qwen2_moe",
            "dbrx",
            "jetmoe",
            "arctic",
        )
        or "num_local_experts" in config_dict
        or "n_routed_experts" in config_dict
    ):
        arch = "moe"
    elif (
        model_type in ("mamba", "mamba2", "falcon_mamba", "ssm")
        or "state_size" in config_dict
        or "d_state" in config_dict
    ):
        arch = "ssm"
    else:
        arch = "transformer"

    # Hidden / Embedding dimension
    embed_dim = (
        config_dict.get("hidden_size")
        or config_dict.get("d_model")
        or config_dict.get("n_embd")
        or config_dict.get("model_dim")
        or 4096
    )

    # Number of layers
    num_layers = (
        config_dict.get("num_hidden_layers")
        or config_dict.get("n_layer")
        or config_dict.get("num_layers")
        or config_dict.get("n_layers")
        or 32
    )

    # Number of query heads
    num_heads = (
        config_dict.get("num_attention_heads")
        or config_dict.get("n_head")
        or config_dict.get("num_heads")
        or 32
    )

    # Number of KV heads (GQA / MQA)
    num_kv_heads = (
        config_dict.get("num_key_value_heads")
        or config_dict.get("n_head_kv")
        or config_dict.get("num_kv_heads")
        or num_heads
    )

    # Intermediate / FFN dimension
    intermediate_dim = (
        config_dict.get("intermediate_size") or config_dict.get("n_inner") or int(embed_dim * 8 / 3)
    )

    # Vocabulary size
    vocab_size = config_dict.get("vocab_size", 32000)

    # Context window length
    max_position = (
        config_dict.get("max_position_embeddings")
        or config_dict.get("n_positions")
        or config_dict.get("seq_length")
        or config_dict.get("max_sequence_length")
    )

    # Numerical dtype
    raw_dtype = str(config_dict.get("torch_dtype", "bfloat16")).lower()
    if "bfloat16" in raw_dtype or "bf16" in raw_dtype:
        dtype = "bf16"
    elif "float16" in raw_dtype or "fp16" in raw_dtype or "half" in raw_dtype:
        dtype = "fp16"
    elif "float32" in raw_dtype or "fp32" in raw_dtype:
        dtype = "fp32"
    elif "int8" in raw_dtype:
        dtype = "int8"
    elif "fp8" in raw_dtype:
        dtype = "fp8"
    else:
        dtype = "fp16"

    # MoE parameters
    num_experts = None
    top_k = None
    if arch == "moe":
        num_experts = (
            config_dict.get("num_local_experts")
            or config_dict.get("num_experts")
            or config_dict.get("n_routed_experts")
            or 8
        )
        top_k = (
            config_dict.get("num_experts_per_tok")
            or config_dict.get("top_k")
            or config_dict.get("num_activated_experts")
            or 2
        )

    # SSM parameters
    state_dim = None
    expand_factor = None
    conv_kernel = None
    if arch == "ssm":
        state_dim = config_dict.get("state_size") or config_dict.get("d_state") or 16
        expand_factor = config_dict.get("expand") or config_dict.get("expand_factor") or 2
        conv_kernel = config_dict.get("conv_kernel") or config_dict.get("d_conv") or 4

    return HFModelConfig(
        model_id=model_id,
        model_type=model_type,
        architecture=arch,
        embed_dim=int(embed_dim),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        intermediate_dim=int(intermediate_dim),
        vocab_size=int(vocab_size),
        max_position_embeddings=int(max_position) if max_position is not None else None,
        dtype=dtype,
        num_experts=int(num_experts) if num_experts is not None else None,
        top_k=int(top_k) if top_k is not None else None,
        state_dim=int(state_dim) if state_dim is not None else None,
        expand_factor=int(expand_factor) if expand_factor is not None else None,
        conv_kernel_size=int(conv_kernel) if conv_kernel is not None else None,
        raw_config=config_dict,
    )


def from_huggingface(
    model_id_or_path: str,
    *,
    revision: str = "main",
    token: str | None = None,
    cache_dir: Path | None = None,
) -> HFModelConfig:
    """Ingest a Hugging Face model and resolve its architectural dimensions directly by model ID.

    Parameters:
        model_id_or_path: Hugging Face hub ID (e.g. 'meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-7B-v0.1')
                          or path to a local directory or config.json.
        revision: Hub revision branch or commit.
        token: Optional HF access token.
        cache_dir: Optional cache directory.
    """
    raw_cfg = fetch_hf_config(model_id_or_path, revision=revision, token=token, cache_dir=cache_dir)
    cleaned_id = model_id_or_path.strip()
    if cleaned_id.lower().startswith("hf:"):
        cleaned_id = cleaned_id[3:].strip()
    return parse_hf_config(raw_cfg, model_id=cleaned_id)
