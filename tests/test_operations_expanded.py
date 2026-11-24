import unittest
from typing import get_args

from neural_cost.estimate import estimate_operation
from neural_cost.operations import Operation, OperationKind


class TestOperationsExpanded(unittest.TestCase):
    def test_softmax_flops(self):
        op = Operation(
            name="softmax_op",
            kind="softmax",
            inputs=((2, 3, 4),),
            output=(2, 3, 4),
            dtype_bytes=4,
        )
        est = estimate_operation(op)
        # 5 * numel(output)
        self.assertEqual(est.flops, 5 * 24)

    def test_layernorm_flops(self):
        op = Operation(
            name="ln_op",
            kind="layernorm",
            inputs=((2, 3, 4), (4,), (4,)),
            output=(2, 3, 4),
            dtype_bytes=4,
        )
        est = estimate_operation(op)
        self.assertEqual(est.flops, 5 * 24)

    def test_batchnorm_flops(self):
        op = Operation(
            name="bn_op",
            kind="batchnorm",
            inputs=((2, 4, 3, 3), (4,), (4,), (4,), (4,)),
            output=(2, 4, 3, 3),
            dtype_bytes=4,
        )
        est = estimate_operation(op)
        self.assertEqual(est.flops, 5 * 72)

    def test_pooling_flops(self):
        op = Operation(
            name="pool_op",
            kind="pooling",
            inputs=((1, 3, 10, 10),),
            output=(1, 3, 5, 5),
            dtype_bytes=2,
            attrs={"kernel_size": (2, 2)},
        )
        est = estimate_operation(op)
        # numel(output) * kH * kW = 75 * 2 * 2 = 300
        self.assertEqual(est.flops, 300)

    def test_pooling_missing_kernel_raises(self):
        op = Operation(
            name="pool_op",
            kind="pooling",
            inputs=((1, 3, 10, 10),),
            output=(1, 3, 5, 5),
            dtype_bytes=2,
            # missing attrs
        )
        with self.assertRaisesRegex(
            ValueError, r"pooling requires attrs\['kernel_size'\] as a tuple"
        ):
            estimate_operation(op)

    def test_all_new_kinds_in_operation_kind(self):
        valid_kinds = get_args(OperationKind)
        self.assertIn("softmax", valid_kinds)
        self.assertIn("layernorm", valid_kinds)
        self.assertIn("batchnorm", valid_kinds)
        self.assertIn("pooling", valid_kinds)
        self.assertIn("rmsnorm", valid_kinds)
        self.assertIn("swiglu", valid_kinds)

    def test_rmsnorm_flops(self):
        op = Operation("rms", "rmsnorm", ((2, 32, 128),), (2, 32, 128), dtype_bytes=4)
        est = estimate_operation(op)
        # 3 * numel(output) = 3 * 8192 = 24576
        self.assertEqual(est.flops, 3 * 2 * 32 * 128)
        self.assertEqual(est.read_bytes, 2 * 32 * 128 * 4)
        self.assertEqual(est.write_bytes, 2 * 32 * 128 * 4)

    def test_swiglu_flops(self):
        op = Operation("swi", "swiglu", ((2, 32, 128),), (2, 32, 128), dtype_bytes=4)
        est = estimate_operation(op)
        # 3 * numel(output) = 24576
        self.assertEqual(est.flops, 3 * 2 * 32 * 128)

    def test_attention_projections_gating(self):
        # 1. Monolithic attention (include_projections=True)
        op_mono = Operation(
            "attn_mono",
            "attention",
            ((2, 32, 128),),
            (2, 32, 128),
            dtype_bytes=4,
            attrs={"num_heads": 4, "seq_len": 32, "head_dim": 32, "include_projections": True},
        )
        est_mono = estimate_operation(op_mono)
        # proj (8388608) + attn core (1089536) = 9478144
        self.assertEqual(est_mono.flops, 9478144)

        # 2. Standalone SDPA kernel (include_projections=False)
        op_sdpa = Operation(
            "sdpa",
            "attention",
            ((2, 4, 32, 32), (2, 4, 32, 32), (2, 4, 32, 32)),
            (2, 4, 32, 32),
            dtype_bytes=4,
            attrs={"num_heads": 4, "seq_len": 32, "head_dim": 32, "include_projections": False},
        )
        est_sdpa = estimate_operation(op_sdpa)
        # attn core only = 1089536
        self.assertEqual(est_sdpa.flops, 1089536)

    def test_fusible_consumer_kinds_contains_rmsnorm_and_swiglu(self):
        from neural_cost.estimate import _FUSIBLE_CONSUMER_KINDS

        self.assertIn("rmsnorm", _FUSIBLE_CONSUMER_KINDS)
        self.assertIn("swiglu", _FUSIBLE_CONSUMER_KINDS)

    def test_read_write_bytes_for_new_ops(self):
        # Softmax
        op = Operation("op1", "softmax", ((10,),), (10,), 4)
        est = estimate_operation(op)
        self.assertEqual(est.read_bytes, 40)
        self.assertEqual(est.write_bytes, 40)

        # LayerNorm
        op = Operation("op2", "layernorm", ((10,), (10,)), (10,), 2)
        est = estimate_operation(op)
        self.assertEqual(est.read_bytes, 40)  # (10 + 10) * 2
        self.assertEqual(est.write_bytes, 20)  # 10 * 2

    def test_invalid_kind_raises(self):
        with self.assertRaisesRegex(ValueError, "unsupported operation kind"):
            Operation("bad", "bogus", ((10,),), (10,))


if __name__ == "__main__":
    unittest.main()
