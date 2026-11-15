"""Inspect V2 config packages without emulating SRAM or device execution.

Core/LUT decoding and subtype1/2/3 shape checks use offline formats; this is not
a general online-config parser. Input SRAM words remain opaque payload bits.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..coordinate import CoordXY, CoordZXYOffset
from ..routing_hexa import AERPacket, AERPacketZXYCopy, aer_packet_walk
from .frame_defs import FFV2, OfflineConfigFrame1FormatV2, OfflineConfigFrame2FormatV2
from .frame_defs import FrameHeader as FH
from .types import FrameArrayLike
from .utils import frame_array2np

__all__ = [
    "AerRouteFields",
    "FrameHeaderInfo",
    "FramePackageInfo",
    "FrameParseError",
    "LutEntry",
    "ParsedCoreFrames",
    "ParsedFrameStream",
    "bit_field",
    "decode_aer_route_fields",
    "decode_core_config",
    "decode_header",
    "decode_lut_entries",
    "expand_aer_destinations",
    "parse_frame_stream",
    "sign_magnitude_to_int",
]

CONFIG1_WORDS = 3
LUT_ENTRIES = 256
SRAM_WORDS = 2


class FrameParseError(ValueError):
    """Malformed package/input, with optional frame_index, raw_frame and context.

    ``reason`` retains the message; supplied diagnostics also appear in str(exc).
    Input normalization can instead raise its original NumPy/Python exception.
    """

    def __init__(
        self,
        message: str,
        *,
        frame_index: int | None = None,
        raw_frame: int | None = None,
        context: dict[str, int | str] | None = None,
    ) -> None:
        self.reason = message
        self.frame_index = frame_index
        self.raw_frame = raw_frame
        self.context = context or {}
        details = []
        if frame_index is not None:
            details.append(f"frame_index={frame_index}")
        if raw_frame is not None:
            details.append(f"raw=0x{raw_frame:016x}")
        details.extend(f"{key}={value}" for key, value in self.context.items())
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(f"{message}{suffix}")


@dataclass(frozen=True)
class FrameHeaderInfo:
    """Common header bits and signed Z/X/Y routing fields.

    frame_type is the 2-bit class; header is the combined 4-bit code. index/value
    retain position/word and payload is low 24 bits. x/y project to (Z+X, Z+Y)
    relative to origin zero; coord_key/coord do not expand multicast copies.
    offset/ncopy expose routing objects, not absolute destinations.
    """

    index: int
    value: int
    frame_type: int
    header: int
    offset_z: int
    offset_x: int
    offset_y: int
    copy_z: int
    copy_x: int
    copy_y: int
    payload: int

    @property
    def x(self) -> int:
        return self.offset_z + self.offset_x

    @property
    def y(self) -> int:
        return self.offset_z + self.offset_y

    @property
    def coord(self) -> CoordXY:
        return CoordXY(self.x, self.y)

    @property
    def coord_key(self) -> tuple[int, int]:
        return self.x, self.y

    @property
    def offset(self) -> CoordZXYOffset:
        return CoordZXYOffset(self.offset_z, self.offset_x, self.offset_y)

    @property
    def ncopy(self) -> AERPacketZXYCopy:
        return AERPacketZXYCopy(self.copy_z, self.copy_x, self.copy_y)


@dataclass(frozen=True)
class FramePackageInfo:
    """Package metadata and raw words, retaining transmission order.

    frame_type is config subtype 1..4; package_type is raw bit23. start_addr has
    subtype-specific units. package_count excludes the header; frame_start/end
    are inclusive indices. header_hex/payload_hex format 16-digit hex words.
    """

    frame_type: int
    start_addr: int
    package_type: int
    package_count: int
    frame_start: int
    frame_end: int
    header: int
    payloads: tuple[int, ...]

    @property
    def header_hex(self) -> str:
        return f"0x{self.header:016x}"

    @property
    def payload_hex(self) -> tuple[str, ...]:
        return tuple(f"0x{value:016x}" for value in self.payloads)


@dataclass
class ParsedCoreFrames:
    """Arrival-ordered packages grouped by projected relative (x, y), not chip ID.

    frame_typeN_payloads concatenate words without applying SRAM addresses or
    repeated writes. Routes with equal projections share a group; copies are
    not expanded. Use packages for original addresses, boundaries and routes.
    """

    x: int
    y: int
    frame_type1_payloads: list[int] = field(default_factory=list)
    frame_type2_payloads: list[int] = field(default_factory=list)
    frame_type3_payloads: list[int] = field(default_factory=list)
    packages: list[FramePackageInfo] = field(default_factory=list)
    frame_type4_payloads: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedFrameStream:
    """Stream word count, arrival-ordered packages and relative-offset core groups."""

    frame_count: int
    packages: list[FramePackageInfo]
    cores: dict[tuple[int, int], ParsedCoreFrames]


@dataclass(frozen=True)
class AerRouteFields:
    """Signed route/copy components; offset/ncopy construct routing objects."""

    offset_z: int
    offset_x: int
    offset_y: int
    copy_z: int
    copy_x: int
    copy_y: int

    @property
    def offset(self) -> CoordZXYOffset:
        return CoordZXYOffset(self.offset_z, self.offset_x, self.offset_y)

    @property
    def ncopy(self) -> AERPacketZXYCopy:
        return AERPacketZXYCopy(self.copy_z, self.copy_x, self.copy_y)


@dataclass(frozen=True)
class LutEntry:
    """Offline LUT index, signed int32 potential, raw uint8 activation and word."""

    index: int
    potential: int
    activation: int
    raw: int


def bit_field(value: int, offset: int, mask: int) -> int:
    """Extract ``(value >> offset) & mask`` using an unshifted mask."""
    return (value >> offset) & mask


def sign_magnitude_to_int(value: int, bits: int = 6) -> int:
    """Decode an already-masked sign-magnitude field; both zero codes give zero."""
    sign = value >> (bits - 1)
    magnitude = value & ((1 << (bits - 1)) - 1)
    return -magnitude if sign else magnitude


def decode_header(frame: int, index: int = 0) -> FrameHeaderInfo:
    """Decode common fields of a header word, retaining the supplied index.

    Call on package headers, not headerless data. No uint64-range, supported
    subtype or package-length validation is performed here.
    """
    header = bit_field(frame, FFV2.GENERAL_HEADER_OFFSET, FFV2.GENERAL_HEADER_MASK)
    return FrameHeaderInfo(
        index=index,
        value=frame,
        frame_type=bit_field(
            frame, FFV2.GENERAL_FRAME_TYPE_OFFSET, FFV2.GENERAL_FRAME_TYPE_MASK
        ),
        header=header,
        offset_z=sign_magnitude_to_int(
            bit_field(
                frame, FFV2.GENERAL_CORE_XY_ADDR_OFFSET, FFV2.GENERAL_CORE_XY_ADDR_MASK
            )
        ),
        offset_x=sign_magnitude_to_int(
            bit_field(
                frame, FFV2.GENERAL_CORE_X_ADDR_OFFSET, FFV2.GENERAL_CORE_X_ADDR_MASK
            )
        ),
        offset_y=sign_magnitude_to_int(
            bit_field(
                frame, FFV2.GENERAL_CORE_Y_ADDR_OFFSET, FFV2.GENERAL_CORE_Y_ADDR_MASK
            )
        ),
        copy_z=sign_magnitude_to_int(
            bit_field(
                frame, FFV2.GENERAL_COPY_XY_ADDR_OFFSET, FFV2.GENERAL_COPY_XY_ADDR_MASK
            )
        ),
        copy_x=sign_magnitude_to_int(
            bit_field(
                frame, FFV2.GENERAL_COPY_X_ADDR_OFFSET, FFV2.GENERAL_COPY_X_ADDR_MASK
            )
        ),
        copy_y=sign_magnitude_to_int(
            bit_field(
                frame, FFV2.GENERAL_COPY_Y_ADDR_OFFSET, FFV2.GENERAL_COPY_Y_ADDR_MASK
            )
        ),
        payload=bit_field(
            frame, FFV2.GENERAL_PAYLOAD_OFFSET, FFV2.GENERAL_PAYLOAD_MASK
        ),
    )


def parse_frame_stream(frames: FrameArrayLike) -> ParsedFrameStream:
    """Parse config subtypes 1..4; retain package order and raw payload words.

    Input normalization follows frame_array2np. FrameParseError reports unsupported
    headers, truncation, subtype1/2 counts other than 3/256, or subtype3 odd counts
    and ranges beyond 4096 two-word records. Type4 geometry is not validated.
    Groups use relative offset projections without copy expansion or write merging.
    Example: ``parse_frame_stream(frames).packages[0].payloads``.
    """
    array = frame_array2np(frames)
    cores: dict[tuple[int, int], ParsedCoreFrames] = {}
    packages: list[FramePackageInfo] = []
    cursor = 0

    while cursor < len(array):
        header = decode_header(int(array[cursor]), cursor)
        if header.header not in {
            FH.CONFIG_TYPE1,
            FH.CONFIG_TYPE2,
            FH.CONFIG_TYPE3,
            FH.CONFIG_TYPE4,
        }:
            raise FrameParseError(
                "unsupported frame header in V2 config stream",
                frame_index=cursor,
                raw_frame=int(array[cursor]),
                context={"header": header.header},
            )

        package_count = bit_field(
            header.payload,
            FFV2.GENERAL_PACKAGE_NUM_OFFSET,
            FFV2.GENERAL_PACKAGE_NUM_MASK,
        )
        frame_end = cursor + package_count
        if frame_end >= len(array):
            raise FrameParseError(
                "truncated V2 config frame package",
                frame_index=cursor,
                raw_frame=int(array[cursor]),
                context={
                    "header": header.header,
                    "package_count": package_count,
                    "frame_count": len(array),
                },
            )

        payloads = tuple(array[cursor + 1 : frame_end + 1].tolist())
        package = FramePackageInfo(
            frame_type=header.header + 1,
            start_addr=bit_field(
                header.payload,
                FFV2.GENERAL_PACKAGE_NEU_START_ADDR_OFFSET,
                FFV2.GENERAL_PACKAGE_NEU_START_ADDR_MASK,
            ),
            package_type=bit_field(
                header.payload,
                FFV2.GENERAL_PACKAGE_TYPE_OFFSET,
                FFV2.GENERAL_PACKAGE_TYPE_MASK,
            ),
            package_count=package_count,
            frame_start=cursor,
            frame_end=frame_end,
            header=int(array[cursor]),
            payloads=payloads,
        )
        _require_package_shape(package)
        packages.append(package)

        coord_key = header.coord_key
        core = cores.get(coord_key)
        if core is None:
            core = ParsedCoreFrames(x=coord_key[0], y=coord_key[1])
            cores[coord_key] = core
        core.packages.append(package)
        if header.header == FH.CONFIG_TYPE1:
            core.frame_type1_payloads.extend(payloads)
        elif header.header == FH.CONFIG_TYPE2:
            core.frame_type2_payloads.extend(payloads)
        elif header.header == FH.CONFIG_TYPE3:
            core.frame_type3_payloads.extend(payloads)
        else:
            core.frame_type4_payloads.extend(payloads)
        cursor = frame_end + 1

    return ParsedFrameStream(frame_count=len(array), packages=packages, cores=cores)


def decode_core_config(payloads: Sequence[int]) -> dict[str, int]:
    """Decode the first three offline subtype1 payload words, ignoring extras.

    Pass ``package.payloads``, without its header; fewer words raises FrameParseError.
    Returned integers are hardware codes, including width/LCN indices and raw
    sign-magnitude test routes, not validated enums or online core parameters.
    """
    if len(payloads) < CONFIG1_WORDS:
        raise FrameParseError(
            "config frame type1 requires at least 3 payload words",
            context={"payload_count": len(payloads)},
        )

    f = OfflineConfigFrame1FormatV2
    w1, w2, w3 = (int(payloads[0]), int(payloads[1]), int(payloads[2]))
    test_core_y = (
        bit_field(w1, f.Word1.TEST_CORE_Y_HIGH2_OFFSET, f.Word1.TEST_CORE_Y_HIGH2_MASK)
        << 4
    ) | bit_field(w2, f.Word2.TEST_CORE_Y_LOW4_OFFSET, f.Word2.TEST_CORE_Y_LOW4_MASK)
    return {
        "snn_ann": bit_field(w1, f.Word1.SNN_ANN_OFFSET, f.Word1.SNN_ANN_MASK),
        "max_pooling": bit_field(
            w1, f.Word1.MAX_POOLING_OFFSET, f.Word1.MAX_POOLING_MASK
        ),
        "add_potential": bit_field(
            w1, f.Word1.ADD_POTENTIAL_OFFSET, f.Word1.ADD_POTENTIAL_MASK
        ),
        "zero_output": bit_field(
            w1, f.Word1.ZERO_OUTPUT_OFFSET, f.Word1.ZERO_OUTPUT_MASK
        ),
        "input_sign": bit_field(w1, f.Word1.INPUT_SIGN_OFFSET, f.Word1.INPUT_SIGN_MASK),
        "input_width": bit_field(
            w1, f.Word1.INPUT_WIDTH_OFFSET, f.Word1.INPUT_WIDTH_MASK
        ),
        "output_sign": bit_field(
            w1, f.Word1.OUTPUT_SIGN_OFFSET, f.Word1.OUTPUT_SIGN_MASK
        ),
        "output_width": bit_field(
            w1, f.Word1.OUTPUT_WIDTH_OFFSET, f.Word1.OUTPUT_WIDTH_MASK
        ),
        "weight_sign": bit_field(
            w1, f.Word1.WEIGHT_SIGN_OFFSET, f.Word1.WEIGHT_SIGN_MASK
        ),
        "weight_width": bit_field(
            w1, f.Word1.WEIGHT_WIDTH_OFFSET, f.Word1.WEIGHT_WIDTH_MASK
        ),
        "lcn": bit_field(w1, f.Word1.LCN_OFFSET, f.Word1.LCN_MASK),
        "target_lcn": bit_field(w1, f.Word1.TARGET_LCN_OFFSET, f.Word1.TARGET_LCN_MASK),
        "axon_skew": bit_field(w1, f.Word1.AXON_SKEW_OFFSET, f.Word1.AXON_SKEW_MASK),
        "neuron_number": bit_field(
            w1, f.Word1.NEURON_NUMBER_OFFSET, f.Word1.NEURON_NUMBER_MASK
        ),
        "test_core_xy": bit_field(
            w1, f.Word1.TEST_CORE_XY_OFFSET, f.Word1.TEST_CORE_XY_MASK
        ),
        "test_core_x": bit_field(
            w1, f.Word1.TEST_CORE_X_OFFSET, f.Word1.TEST_CORE_X_MASK
        ),
        "test_core_y": test_core_y,
        "global_send": bit_field(
            w2, f.Word2.GLOBAL_SEND_OFFSET, f.Word2.GLOBAL_SEND_MASK
        ),
        "csc_accelerate": bit_field(
            w2, f.Word2.CSC_ACCELERATE_OFFSET, f.Word2.CSC_ACCELERATE_MASK
        ),
        "global_receive": bit_field(
            w2, f.Word2.GLOBAL_RECEIVE_OFFSET, f.Word2.GLOBAL_RECEIVE_MASK
        ),
        "thread_number": bit_field(
            w2, f.Word2.THREAD_NUMBER_OFFSET, f.Word2.THREAD_NUMBER_MASK
        ),
        "busy_cycle": bit_field(w2, f.Word2.BUSY_CYCLE_OFFSET, f.Word2.BUSY_CYCLE_MASK),
        "delay_cycle": bit_field(
            w2, f.Word2.DELAY_CYCLE_OFFSET, f.Word2.DELAY_CYCLE_MASK
        ),
        "width_cycle": bit_field(
            w2, f.Word2.WIDTH_CYCLE_OFFSET, f.Word2.WIDTH_CYCLE_MASK
        ),
        "tick_start": bit_field(w3, f.Word3.TICK_START_OFFSET, f.Word3.TICK_START_MASK),
        "tick_duration": bit_field(
            w3, f.Word3.TICK_DURATION_OFFSET, f.Word3.TICK_DURATION_MASK
        ),
        "tick_initial": bit_field(
            w3, f.Word3.TICK_INITIAL_OFFSET, f.Word3.TICK_INITIAL_MASK
        ),
    }


def decode_lut_entries(payloads: list[int] | tuple[int, ...]) -> list[LutEntry]:
    """Decode the first 256 offline subtype2 payload words; ignore extras.

    Fewer words raises FrameParseError. Potentials use int32 two's complement;
    activations remain unsigned byte codes. This does not decode online FP bits.
    """
    if len(payloads) < LUT_ENTRIES:
        raise FrameParseError(
            "config frame type2 LUT requires at least 256 payload words",
            context={"payload_count": len(payloads)},
        )

    f = OfflineConfigFrame2FormatV2
    return [
        LutEntry(
            index=index,
            potential=twos_complement_to_int(
                bit_field(value, f.POTENTIAL_OFFSET, f.POTENTIAL_MASK), 32
            ),
            activation=bit_field(value, f.ACTIVATION_OFFSET, f.ACTIVATION_MASK),
            raw=value,
        )
        for index, value in enumerate(payloads[:LUT_ENTRIES])
    ]


def decode_aer_route_fields(
    addr_core_xy: int,
    addr_core_x: int,
    addr_core_y: int,
    addr_copy_xy: int,
    addr_copy_x: int,
    addr_copy_y: int,
) -> AerRouteFields:
    """Decode masked 6-bit core XY/X/Y and copy XY/X/Y sign-magnitude fields.

    XY denotes diagonal Z. Inputs are raw codes, not signed coordinates;
    no extra width/range validation is performed.
    """
    return AerRouteFields(
        offset_z=sign_magnitude_to_int(addr_core_xy),
        offset_x=sign_magnitude_to_int(addr_core_x),
        offset_y=sign_magnitude_to_int(addr_core_y),
        copy_z=sign_magnitude_to_int(addr_copy_xy),
        copy_x=sign_magnitude_to_int(addr_copy_x),
        copy_y=sign_magnitude_to_int(addr_copy_y),
    )


def expand_aer_destinations(
    source: CoordXY | tuple[int, int], route: AerRouteFields
) -> list[CoordXY]:
    """Expand a decoded route from an explicit source into absolute destinations.

    The source is copied, then the shared AER walker supplies ordering and
    coordinate checks. Example: ``expand_aer_destinations((0, 0), route)``.
    Packet source and board topology cannot be inferred from the route alone.
    """
    source_coord = source.copy() if isinstance(source, CoordXY) else CoordXY(*source)
    packet = AERPacket(source_coord, route.offset, route.ncopy)
    return aer_packet_walk(packet)


def twos_complement_to_int(value: int, bits: int) -> int:
    """Interpret low bits as two's complement; require a positive width."""
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    value &= mask
    return value - (1 << bits) if value & sign_bit else value


def _require_package_shape(package: FramePackageInfo) -> None:
    if package.frame_end != package.frame_start + package.package_count:
        raise FrameParseError(
            "truncated V2 config frame package",
            frame_index=package.frame_start,
            raw_frame=package.header,
            context={
                "frame_type": package.frame_type,
                "expected_end": package.frame_start + package.package_count,
                "actual_end": package.frame_end,
            },
        )
    if package.frame_type == 1 and package.package_count != CONFIG1_WORDS:
        raise FrameParseError(
            "config frame type1 must contain exactly 3 payload words",
            frame_index=package.frame_start,
            raw_frame=package.header,
            context={"package_count": package.package_count},
        )
    if package.frame_type == 2 and package.package_count != LUT_ENTRIES:
        raise FrameParseError(
            "config frame type2 LUT must contain exactly 256 entries",
            frame_index=package.frame_start,
            raw_frame=package.header,
            context={"package_count": package.package_count},
        )
    if package.frame_type == 3 and package.package_count % SRAM_WORDS:
        raise FrameParseError(
            "config frame type3 payload must contain whole 128-bit SRAM records",
            frame_index=package.frame_start,
            raw_frame=package.header,
            context={"package_count": package.package_count},
        )
    if (
        package.frame_type == 3
        and package.start_addr + package.package_count // SRAM_WORDS > 4096
    ):
        raise FrameParseError(
            "config frame type3 SRAM address range exceeds 4096 records",
            frame_index=package.frame_start,
            raw_frame=package.header,
            context={
                "start_addr": package.start_addr,
                "package_count": package.package_count,
            },
        )
