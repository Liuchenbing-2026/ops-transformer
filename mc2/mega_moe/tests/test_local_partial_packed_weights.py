"""CPU rejection checks for the opt-in TP4 packed BF16 extension entry."""

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture(scope="module")
def compiled_mega_moe():
    from cann_ops_transformer.ops.mega_moe import _mega_moe_op_builder

    assert not torch.npu.is_initialized()
    module = _mega_moe_op_builder.load()
    assert not torch.npu.is_initialized()
    return module


@pytest.mark.parametrize(
    "invalid",
    ["expert_count", "down_shape", "dtype", "mixed_lists", "multiple_packed", "strided", "dtype_override"],
)
def test_invalid_packed_weights_reject_before_npu_init(compiled_mega_moe, invalid):
    w1 = torch.empty((64, 2048, 1024), dtype=torch.bfloat16, device="meta")
    w2 = torch.empty((64, 512, 2048), dtype=torch.bfloat16, device="meta")
    weights1, weights2 = [w1], [w2]
    weight1_type = None
    if invalid == "expert_count":
        weights1 = [w1[:63]]
    elif invalid == "down_shape":
        weights2 = [w2[..., :1024]]
    elif invalid == "dtype":
        weights1 = [w1.float()]
    elif invalid == "mixed_lists":
        weights2 = list(w2.unbind())
    elif invalid == "multiple_packed":
        weights1 = [w1, w1]
    elif invalid == "strided":
        weights1 = [torch.empty((64, 2048, 2048), dtype=torch.bfloat16, device="meta")[..., ::2]]
    elif invalid == "dtype_override":
        weight1_type = 0

    x = torch.empty((1, 2048), dtype=torch.bfloat16)
    ids = torch.zeros((1, 8), dtype=torch.int32)
    probabilities = torch.ones((1, 8), dtype=torch.float32)
    args = [torch.empty(1), x, ids, probabilities, weights1, weights2, 256, 4, 1050 * 1024 * 1024]
    args += [None] * 11
    args += [65792, 0, 0, "local_partial_tp4", 8224, "swiglu", [3.4028234663852886e38]]
    args += [None, weight1_type, None, None, None, 0]
    assert not torch.npu.is_initialized()
    with pytest.raises(RuntimeError, match="Packed local_partial_tp4 requires contiguous BF16 ND weights"):
        compiled_mega_moe.npu_mega_moe(*args)
    assert not torch.npu.is_initialized()
