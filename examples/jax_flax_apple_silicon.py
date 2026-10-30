"""Profile a Flax ResNet on Apple Silicon: CPU vs GPU (jax-mps / jax-metal).

An AlphaZero-style policy/value net (Flax ``linen`` ResNet with GroupNorm) on a
batch of 8x8x119 board encodings, profiled end to end with neural-cost:

1. Pick the JAX platform before importing JAX::

       JAX_PLATFORMS=cpu python examples/jax_flax_apple_silicon.py
       JAX_PLATFORMS=mps python examples/jax_flax_apple_silicon.py    # jax-mps
       JAX_PLATFORMS=METAL python examples/jax_flax_apple_silicon.py  # jax-metal

2. Or compare platforms in one table (each one runs in its own subprocess)::

       python examples/jax_flax_apple_silicon.py --compare cpu,mps

Requirements: ``pip install "neural-cost[jax]" flax``.  For the GPU, also
install ``"neural-cost[jax-mps]"`` (jax-mps 0.11.x needs jax/jaxlib 0.11.x and
Python >= 3.12) or ``"neural-cost[jax-metal]"``.
"""

from __future__ import annotations

import argparse
import os
import sys

BATCH, BOARD, PLANES = 128, 8, 119


def build(channels: int = 64, blocks: int = 4, batch: int = BATCH):
    """Return ``(apply_fn, (variables, obs))``.  Also the target for ``--compare``."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    class Block(nn.Module):
        channels: int

        @nn.compact
        def __call__(self, x):
            y = nn.relu(nn.GroupNorm(num_groups=8)(nn.Conv(self.channels, (3, 3))(x)))
            y = nn.GroupNorm(num_groups=8)(nn.Conv(self.channels, (3, 3))(y))
            return nn.relu(x + y)

    class ResNet(nn.Module):
        channels: int
        blocks: int

        @nn.compact
        def __call__(self, x):
            x = nn.relu(nn.GroupNorm(num_groups=8)(nn.Conv(self.channels, (3, 3))(x)))
            for _ in range(self.blocks):
                x = Block(self.channels)(x)
            policy = nn.Dense(4672)(x.reshape(x.shape[0], -1))
            value = jnp.tanh(nn.Dense(1)(x.mean(axis=(1, 2))))
            return policy, value

    model = ResNet(channels, blocks)
    obs = jnp.ones((batch, BOARD, BOARD, PLANES), jnp.float32)
    variables = model.init(jax.random.key(0), obs)
    # Jit model.apply once and pass the weights as an argument: JaxAdapter then
    # treats argument 0 as parameters (parameter_argnums=(0,)).
    return jax.jit(model.apply), (variables, obs)


def profile() -> None:
    import jax

    from neural_cost import analyze_gap, estimate_operations
    from neural_cost.adapters import JaxAdapter

    device = jax.devices()[0]
    print(f"JAX {jax.__version__} on {device.platform} ({device})")

    apply_fn, (variables, obs) = build()
    adapter = JaxAdapter(parameter_argnums=(0,))

    # Device-aware spec: the CPU is scored against the CPU, MPS against the GPU.
    # Backends without a spec-table entry are measured on the device instead.
    hardware, _ = adapter.hardware(device, benchmark_memory=False)
    print(f"hardware: {hardware.name} [{hardware.device_type}] "
          f"{hardware.peak_flops / 1e12:.2f} TFLOP/s, {hardware.memory_bandwidth / 1e9:.0f} GB/s")

    operations = adapter.operations(apply_fn, (variables, obs))
    measurement = adapter.benchmark(apply_fn, variables, obs, warmup=5, repeats=20)
    gap = analyze_gap(
        estimate_operations(operations),
        measurement,
        hardware,
        operations=operations,
        coverage=adapter.coverage,
    )
    # Grouping by Flax module scope shows whether GroupNorm or Conv dominates.
    print(gap.render(top=8, group_by_scope=True))
    print(f"(batch {BATCH}, {len(measurement.samples_seconds)} timed runs, device {measurement.device})")


def compare(platforms: str) -> None:
    from neural_cost import compare_backends

    target = f"{os.path.abspath(__file__)}:build"
    print(compare_backends(target, platforms.split(","), warmup=5, repeats=20).render())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare", metavar="PLATFORMS", help="e.g. cpu,mps or cpu,METAL")
    args = parser.parse_args()
    try:
        import flax  # noqa: F401
    except ImportError:
        sys.exit("This example needs Flax: pip install flax")
    compare(args.compare) if args.compare else profile()
