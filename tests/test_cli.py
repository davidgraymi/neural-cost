"""Unit and integration tests for the unified neural-cost CLI suite."""

from __future__ import annotations

import json

import pytest

from neural_cost.cli import build_parser, compare_main, main
from neural_cost.cli.helpers import (
    format_bytes,
    format_flops,
    parse_bytes,
    parse_dtype_bytes,
    resolve_hardware,
)


class TestCliHelpers:
    """Test CLI string parsing and formatting utilities."""

    def test_parse_bytes(self):
        assert parse_bytes(1024) == 1024
        assert parse_bytes(2048.0) == 2048
        assert parse_bytes("1024") == 1024
        assert parse_bytes("1KB") == 1024
        assert parse_bytes("16GB") == 16 * 1024**3
        assert parse_bytes("512MB") == 512 * 1024**2
        assert parse_bytes("1.5 TB") == int(1.5 * 1024**4)
        assert parse_bytes("4G") == 4 * 1024**3

        with pytest.raises(ValueError, match="Cannot parse byte size"):
            parse_bytes("invalid_size")
        with pytest.raises(ValueError, match="Unknown byte unit"):
            parse_bytes("100xyz")
        with pytest.raises(ValueError, match="Empty byte size"):
            parse_bytes("   ")

    def test_parse_dtype_bytes(self):
        assert parse_dtype_bytes("fp32") == 4
        assert parse_dtype_bytes("fp16") == 2
        assert parse_dtype_bytes("bf16") == 2
        assert parse_dtype_bytes("int8") == 1
        assert parse_dtype_bytes("int4") == 1

        with pytest.raises(ValueError, match="Unsupported dtype"):
            parse_dtype_bytes("complex128")

    def test_format_bytes(self):
        assert "1.00 KB" == format_bytes(1024)
        assert "16.00 GB" == format_bytes(16 * 1024**3)
        assert "512 B" == format_bytes(512)

    def test_format_flops(self):
        assert "2.00 GFLOP" == format_flops(2e9)
        assert "989.00 TFLOP" == format_flops(989e12)

    def test_resolve_hardware(self):
        hw, det = resolve_hardware(
            peak_flops=1e12,
            memory_bandwidth=100e9,
            device_name="TestDevice",
        )
        assert hw.device_name == "TestDevice"
        assert hw.peak_flops == 1e12
        assert hw.memory_bandwidth == 100e9
        assert det is None

        # Auto-detect without STREAM benchmark
        hw_auto, det_auto = resolve_hardware(benchmark_memory=False)
        assert hw_auto.peak_flops > 0
        assert hw_auto.memory_bandwidth > 0
        assert det_auto is not None


class TestRootCLI:
    """Test top-level CLI command parsing and routing."""

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        out, _ = capsys.readouterr()
        assert "neural-cost" in out

    def test_no_arguments_prints_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 0
        out, _ = capsys.readouterr()
        assert "Available neural-cost subcommands" in out

    def test_build_parser(self):
        parser = build_parser()
        assert parser.prog == "neural-cost"


class TestHardwareCommand:
    """Test neural-cost hardware subcommand."""

    def test_hardware_text(self, capsys):
        main(["hardware", "--no-bench"])
        out, _ = capsys.readouterr()
        assert "Detected Hardware Specification" in out
        assert "Roofline Ridge" in out

    def test_hardware_json(self, capsys):
        main(["hardware", "--no-bench", "--json"])
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert "device_name" in data
        assert "peak_flops" in data
        assert "memory_bandwidth_bytes_per_sec" in data
        assert "ridge_point_flop_per_byte" in data


class TestProfileCommand:
    """Test neural-cost profile across model architectures."""

    @pytest.mark.parametrize(
        "arch",
        ["transformer", "moe", "paged-attention", "mlp", "convnet"],
    )
    def test_profile_json(self, arch, capsys):
        main(
            [
                "profile",
                "--arch",
                arch,
                "--batch-size",
                "2",
                "--no-bench",
                "--json",
            ]
        )
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert data["architecture"] == arch
        assert "flops" in data
        assert "total_bytes" in data
        assert "arithmetic_intensity" in data
        assert "roofline" in data
        assert data["roofline"]["bottleneck"] in ("compute", "memory")

    def test_profile_text_table(self, capsys):
        main(["profile", "--arch", "transformer", "--no-bench"])
        out, _ = capsys.readouterr()
        assert "Neural Cost Static Profile" in out
        assert "Roofline Projection" in out
        assert "Lower-Bound Latency" in out

    def test_profile_hf_model_positional(self, tmp_path, capsys):
        cfg = {
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 5632,
            "torch_dtype": "bfloat16",
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        main(["profile", str(cfg_path), "--no-bench", "--json"])
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert data["architecture"] == "transformer"
        assert data["embed_dim"] == 2048
        assert data["num_layers"] == 16
        assert data["num_heads"] == 16
        assert data["num_kv_heads"] == 4
        assert data["dtype"] == "bf16"

    def test_profile_svg_plot_export(self, tmp_path, capsys):
        plot_file = tmp_path / "test_roofline.svg"
        main(
            [
                "profile",
                "--arch",
                "transformer",
                "--plot",
                str(plot_file),
                "--plot-theme",
                "light",
                "--no-bench",
                "--json",
            ]
        )
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert "plot_path" in data
        assert plot_file.is_file()
        svg_content = plot_file.read_text(encoding="utf-8")
        assert svg_content.startswith("<svg")
        assert 'fill="#ffffff"' in svg_content


class TestAuditCommand:
    """Test neural-cost audit CI assertions and budget gates."""

    def test_audit_pass(self, capsys):
        # A tiny model well within a 10 GB VRAM and 1000ms latency budget
        main(
            [
                "audit",
                "--arch",
                "transformer",
                "--batch-size",
                "1",
                "--seq-len",
                "128",
                "--embed-dim",
                "512",
                "--num-heads",
                "8",
                "--num-layers",
                "4",
                "--max-vram",
                "10GB",
                "--max-latency-ms",
                "1000.0",
                "--no-bench",
            ]
        )
        out, _ = capsys.readouterr()
        assert "AUDIT PASSED" in out
        assert "[✓ PASS] VRAM Footprint" in out
        assert "[✓ PASS] Latency Floor" in out

    def test_audit_fail(self, capsys):
        # Intentionally impossible 10 KB budget for a transformer
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "audit",
                    "--arch",
                    "transformer",
                    "--batch-size",
                    "1",
                    "--seq-len",
                    "1024",
                    "--embed-dim",
                    "1024",
                    "--num-heads",
                    "16",
                    "--num-layers",
                    "8",
                    "--max-vram",
                    "10KB",
                    "--no-bench",
                ]
            )
        assert exc.value.code == 1
        out, err = capsys.readouterr()
        assert "[✗ FAIL] VRAM Footprint" in out
        assert "AUDIT FAILED" in err

    def test_audit_json_output(self, capsys):
        main(
            [
                "audit",
                "--arch",
                "mlp",
                "--in-features",
                "256",
                "--out-features",
                "256",
                "--max-vram",
                "1GB",
                "--no-bench",
                "--json",
            ]
        )
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert data["status"] == "PASS"
        assert len(data["checks"]) == 1
        assert data["checks"][0]["passed"] is True

    def test_audit_hf_model(self, tmp_path, capsys):
        cfg = {
            "model_type": "llama",
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
            "torch_dtype": "bfloat16",
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        main(
            [
                "audit",
                "--model",
                str(cfg_path),
                "--max-vram",
                "10GB",
                "--no-bench",
                "--json",
            ]
        )
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert data["status"] == "PASS"


class TestLLMCommand:
    """Test neural-cost llm dedicated subcommands."""

    def test_llm_moe(self, capsys):
        main(["llm", "moe", "--json"])
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert "total_parameters" in data
        assert "active_parameters" in data
        assert "expected_loaded_experts" in data
        assert data["bottleneck"] in ("compute", "memory")

    def test_llm_speculative(self, capsys):
        main(["llm", "speculative", "--gamma", "4", "--acceptance-rate", "0.8", "--json"])
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert data["gamma"] == 4
        assert "expected_tokens_per_step" in data
        assert "speedup" in data
        assert "breakeven_acceptance_rate" in data

    def test_llm_paged(self, capsys):
        main(["llm", "paged", "--batch-size", "16", "--context-len", "1024", "--json"])
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert "allocated_kv_bytes" in data
        assert "memory_saved_bytes" in data
        assert "concurrency_multiplier" in data

    def test_llm_continuous(self, capsys):
        main(["llm", "continuous", "--decode-streams", "32", "--prefill-tokens", "256", "--json"])
        out, _ = capsys.readouterr()
        data = json.loads(out)
        assert data["total_tokens"] == 32 + 256
        assert "arithmetic_intensity" in data
        assert "bottleneck" in data


class TestBackwardCompatibility:
    """Test backward compatibility of neural-cost-compare and _cli module."""

    def test_compare_cli_help(self, capsys):
        with pytest.raises(SystemExit) as exc:
            compare_main(["--help"])
        assert exc.value.code == 0
        out, _ = capsys.readouterr()
        assert "Compare framework performance against hardware roofline" in out

    def test_legacy_module_exports(self):
        import neural_cost._cli as legacy

        assert hasattr(legacy, "main")
        assert hasattr(legacy, "run_torch")
        assert hasattr(legacy, "run_jax")
        assert hasattr(legacy, "run_tensorflow")
        assert hasattr(legacy, "Result")
        assert hasattr(legacy, "SHAPES")
