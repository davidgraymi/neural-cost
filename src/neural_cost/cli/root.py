"""Root CLI entrypoint and command router for neural-cost."""

from __future__ import annotations

import argparse
import sys

from neural_cost import __version__
from neural_cost.cli.audit_cmd import register_audit_parser
from neural_cost.cli.benchmark_cmd import register_benchmark_parser
from neural_cost.cli.compare_cmd import register_compare_parser
from neural_cost.cli.hardware_cmd import register_hardware_parser
from neural_cost.cli.llm_cmd import register_llm_parser
from neural_cost.cli.profile_cmd import register_profile_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neural-cost",
        description="Unified theoretical and empirical performance modeling suite for deep learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"neural-cost {__version__}",
        help="Show neural-cost version and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available neural-cost subcommands:",
    )

    register_hardware_parser(subparsers)
    register_profile_parser(subparsers)
    register_benchmark_parser(subparsers)
    register_compare_parser(subparsers)
    register_audit_parser(subparsers)
    register_llm_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    try:
        code = args.func(args)
        if isinstance(code, int) and code != 0:
            sys.exit(code)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted by user.\n")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
