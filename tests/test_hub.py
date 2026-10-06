"""Unit tests for zero-code Hugging Face model configuration ingestion."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from neural_cost.hub import (
    HFModelConfig,
    fetch_hf_config,
    from_huggingface,
    get_default_cache_dir,
    parse_hf_config,
)


@pytest.fixture
def llama3_config_dict() -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "torch_dtype": "bfloat16",
    }


@pytest.fixture
def mixtral_config_dict() -> dict:
    return {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 32000,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "torch_dtype": "float16",
    }


@pytest.fixture
def mamba_config_dict() -> dict:
    return {
        "architectures": ["MambaForCausalLM"],
        "model_type": "mamba",
        "d_model": 2048,
        "n_layer": 48,
        "state_size": 16,
        "expand": 2,
        "conv_kernel": 4,
        "vocab_size": 50277,
        "torch_dtype": "float16",
    }


class TestParseHFConfig:
    def test_parse_llama3(self, llama3_config_dict):
        cfg = parse_hf_config(llama3_config_dict, model_id="meta-llama/Meta-Llama-3-8B")
        assert isinstance(cfg, HFModelConfig)
        assert cfg.model_id == "meta-llama/Meta-Llama-3-8B"
        assert cfg.architecture == "transformer"
        assert cfg.embed_dim == 4096
        assert cfg.num_layers == 32
        assert cfg.num_heads == 32
        assert cfg.num_kv_heads == 8
        assert cfg.intermediate_dim == 14336
        assert cfg.vocab_size == 128256
        assert cfg.max_position_embeddings == 8192
        assert cfg.dtype == "bf16"
        assert cfg.num_experts is None

    def test_parse_mixtral_moe(self, mixtral_config_dict):
        cfg = parse_hf_config(mixtral_config_dict, model_id="mistralai/Mixtral-8x7B-v0.1")
        assert cfg.architecture == "moe"
        assert cfg.embed_dim == 4096
        assert cfg.num_layers == 32
        assert cfg.num_heads == 32
        assert cfg.num_kv_heads == 8
        assert cfg.num_experts == 8
        assert cfg.top_k == 2
        assert cfg.dtype == "fp16"

    def test_parse_mamba_ssm(self, mamba_config_dict):
        cfg = parse_hf_config(mamba_config_dict, model_id="state-spaces/mamba-1.4b")
        assert cfg.architecture == "ssm"
        assert cfg.embed_dim == 2048
        assert cfg.num_layers == 48
        assert cfg.state_dim == 16
        assert cfg.expand_factor == 2
        assert cfg.conv_kernel_size == 4

    def test_parse_fallback_defaults(self):
        minimal = {"model_type": "custom"}
        cfg = parse_hf_config(minimal, model_id="minimal")
        assert cfg.architecture == "transformer"
        assert cfg.embed_dim == 4096
        assert cfg.num_layers == 32
        assert cfg.num_heads == 32
        assert cfg.num_kv_heads == 32
        assert cfg.intermediate_dim == int(4096 * 8 / 3)


class TestFetchHFConfig:
    def test_fetch_from_local_file(self, tmp_path, llama3_config_dict):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(llama3_config_dict), encoding="utf-8")

        res = fetch_hf_config(str(cfg_file))
        assert res["hidden_size"] == 4096
        assert res["num_hidden_layers"] == 32

    def test_fetch_from_local_directory(self, tmp_path, llama3_config_dict):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(llama3_config_dict), encoding="utf-8")

        res = fetch_hf_config(str(tmp_path))
        assert res["hidden_size"] == 4096

    def test_fetch_from_cache(self, tmp_path, llama3_config_dict):
        cache_dir = tmp_path / "hub_cache"
        model_cache = cache_dir / "meta-llama--Meta-Llama-3-8B"
        model_cache.mkdir(parents=True)
        (model_cache / "main_config.json").write_text(
            json.dumps(llama3_config_dict), encoding="utf-8"
        )

        res = fetch_hf_config("meta-llama/Meta-Llama-3-8B", cache_dir=cache_dir)
        assert res["hidden_size"] == 4096

    def test_remote_fetch_and_cache(self, tmp_path, llama3_config_dict):
        cache_dir = tmp_path / "hub_cache"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(llama3_config_dict).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            res = fetch_hf_config(
                "hf:meta-llama/Meta-Llama-3-8B", cache_dir=cache_dir, token="test_token"
            )
            assert res["hidden_size"] == 4096
            mock_urlopen.assert_called_once()
            # Verify request authorization header
            req = mock_urlopen.call_args[0][0]
            assert req.headers["Authorization"] == "Bearer test_token"

        # Verify cached to disk
        cached_file = cache_dir / "meta-llama--Meta-Llama-3-8B" / "main_config.json"
        assert cached_file.is_file()

    def test_remote_404_not_found(self, tmp_path):
        cache_dir = tmp_path / "hub_cache"
        http_err = urllib.error.HTTPError(
            url="http://test", code=404, msg="Not Found", hdrs={}, fp=None
        )
        with (
            patch("urllib.request.urlopen", side_effect=http_err),
            pytest.raises(FileNotFoundError, match="Model or config not found"),
        ):
            fetch_hf_config("nonexistent/model", cache_dir=cache_dir)

    def test_remote_401_gated_or_unauthorized(self, tmp_path):
        cache_dir = tmp_path / "hub_cache"
        http_err = urllib.error.HTTPError(
            url="http://test", code=401, msg="Unauthorized", hdrs={}, fp=None
        )
        with (
            patch("urllib.request.urlopen", side_effect=http_err),
            pytest.raises(PermissionError, match="Access denied"),
        ):
            fetch_hf_config("meta-llama/Meta-Llama-3-8B", cache_dir=cache_dir)


class TestFromHuggingFace:
    def test_from_huggingface_with_file(self, tmp_path, llama3_config_dict):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(llama3_config_dict), encoding="utf-8")

        model_cfg = from_huggingface(str(cfg_file))
        assert model_cfg.architecture == "transformer"
        assert model_cfg.embed_dim == 4096
        assert model_cfg.num_kv_heads == 8

    def test_default_cache_dir(self, monkeypatch):
        monkeypatch.setenv("NEURAL_COST_CACHE_DIR", "/custom/cache/dir")
        p = get_default_cache_dir()
        assert str(p) == "/custom/cache/dir"
