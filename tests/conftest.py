import os

# Ensure JAX uses CPU platform by default during test execution to avoid
# experimental jax-metal plugin incompatibilities on macOS.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
