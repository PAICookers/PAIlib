import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from paicorelib.float_codec import (
    BF16Param,
    FP32Param,
    cast_bf16_scalar,
    cast_fp32_scalar,
    pack_bf16_payload_bits,
    pack_bf16_scalar_bits,
    pack_fp32_payload_bits,
    pack_fp32_scalar_bits,
)

_UINT16_DTYPE = np.dtype("<u2")
_UINT32_DTYPE = np.dtype("<u4")


def pack_fp32_array_bits(values) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(
        _UINT32_DTYPE
    )


def pack_bf16_array_bits(values) -> np.ndarray:
    return (pack_fp32_array_bits(values) >> np.uint32(16)).astype(_UINT16_DTYPE)


def test_fp32_scalar_and_payload_bits():
    values = np.array([[1.0 / 3.0, -0.1], [2.5, np.float32(-4.75)]])
    scalar = values[0, 0]
    int_payload = np.array([-128, -1, 0, 1, 127], dtype=np.int32)

    assert cast_fp32_scalar(scalar) == np.float32(scalar).item()
    assert pack_fp32_scalar_bits(scalar) == int(pack_fp32_array_bits([scalar])[0])
    assert np.array_equal(pack_fp32_payload_bits(values), pack_fp32_array_bits(values))
    assert np.array_equal(
        pack_fp32_payload_bits(int_payload),
        np.ascontiguousarray(int_payload, _UINT32_DTYPE),
    )


def test_bf16_scalar_and_payload_bits():
    values = np.array([[0.1, -0.75], [1.0 / 3.0, 8.0]], dtype=np.float32)
    scalar = np.float32(-0.75)

    assert cast_bf16_scalar(scalar) == cast_fp32_scalar(scalar)
    assert pack_bf16_scalar_bits(scalar) == int(pack_bf16_array_bits([scalar])[0])
    assert np.array_equal(pack_bf16_payload_bits(values), pack_bf16_array_bits(values))


def test_bf16_payload_bits_accept_integer_arrays_as_raw_payload():
    payload = np.array([[0x3F80, 0xC020], [0, 0x4120]], dtype=np.uint16)

    assert np.array_equal(pack_bf16_payload_bits(payload), payload)


def test_fp32_payload_bits_preserve_integer_payloads():
    values = np.array([[-128, -1, 0, 1, 127]], dtype=np.int32)

    assert np.array_equal(pack_fp32_payload_bits(values).view(np.int32), values)


class FloatParams(BaseModel):
    bf16: BF16Param
    fp32: FP32Param


def test_bf16_and_fp32_param_preserve_fp32_carriers():
    params = FloatParams.model_validate(
        {"bf16": np.float32(0.1), "fp32": np.float64(1.0 / 3.0)}, strict=True
    )

    assert params.bf16 == cast_fp32_scalar(np.float32(0.1))
    assert params.fp32 == cast_fp32_scalar(np.float64(1.0 / 3.0))


@pytest.mark.parametrize("field", ["bf16", "fp32"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_float_params_reject_non_finite_values(field, value):
    data = {"bf16": np.float32(0.0), "fp32": np.float32(0.0)}
    data[field] = value

    with pytest.raises(ValidationError):
        FloatParams.model_validate(data, strict=True)
