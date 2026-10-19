import pytest

from neural_cost.adapters import (
    JaxAdapter,
    TensorFlowAdapter,
    TorchAdapter,
    available_adapters,
    get_adapter,
)


def test_get_adapter_torch():
    pytest.importorskip("torch")
    adapter = get_adapter("torch")
    assert isinstance(adapter, TorchAdapter)


def test_get_adapter_pytorch_alias():
    pytest.importorskip("torch")
    adapter = get_adapter("pytorch")
    assert isinstance(adapter, TorchAdapter)


def test_get_adapter_torch_fx():
    pytest.importorskip("torch")
    adapter = get_adapter("torch-fx")
    assert isinstance(adapter, TorchAdapter)
    assert adapter.name == "pytorch-fx"


def test_get_adapter_jax():
    pytest.importorskip("jax")
    adapter = get_adapter("jax")
    assert isinstance(adapter, JaxAdapter)


def test_get_adapter_tensorflow():
    pytest.importorskip("tensorflow")
    adapter = get_adapter("tensorflow")
    assert isinstance(adapter, TensorFlowAdapter)


def test_get_adapter_tf_alias():
    pytest.importorskip("tensorflow")
    adapter = get_adapter("tf")
    assert isinstance(adapter, TensorFlowAdapter)


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError):
        get_adapter("unknown")


def test_available_adapters_returns_list():
    adapters = available_adapters()
    assert isinstance(adapters, list)
    assert all(isinstance(a, str) for a in adapters)
