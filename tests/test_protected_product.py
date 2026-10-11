from __future__ import annotations

import hashlib
import struct
import zlib
from io import BytesIO

import pytest
from PIL import Image

from product_scene_editor.protected_product import compose_protected_product


def png(mode: str, size: tuple[int, int], pixels: bytes, image_format: str = "PNG") -> bytes:
    image = Image.frombytes(mode, size, pixels)
    stream = BytesIO()
    image.save(stream, format=image_format)
    return stream.getvalue()


def valid_inputs() -> tuple[bytes, bytes, bytes, bytes]:
    source = png("RGB", (3, 2), bytes(range(18)))
    background = png("RGB", (3, 2), bytes(range(100, 118)))
    alpha = png("L", (3, 2), bytes((255, 0, 128, 255, 64, 255)))
    protected = png("L", (3, 2), bytes((255, 0, 0, 255, 0, 0)))
    return source, background, alpha, protected


def png_with_dimensions(data: bytes, width: int, height: int) -> bytes:
    ihdr = data[16:29]
    resized = struct.pack(">II", width, height) + ihdr[8:]
    crc = zlib.crc32(b"IHDR" + resized)
    return data[:8] + struct.pack(">I", 13) + b"IHDR" + resized + struct.pack(">I", crc) + data[33:]


def test_composes_background_and_preserves_every_protected_pixel() -> None:
    source, background, alpha, protected = valid_inputs()
    original_inputs = (source, background, alpha, protected)

    output, report = compose_protected_product(*original_inputs)

    with Image.open(BytesIO(output)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.mode == "RGB"
        assert decoded.size == (3, 2)
        result = decoded.tobytes()
    source_pixels = bytes(range(18))
    background_pixels = bytes(range(100, 118))
    alpha_pixels = (255, 0, 128, 255, 64, 255)
    for index, amount in enumerate(alpha_pixels):
        for channel in range(3):
            expected = (
                source_pixels[index * 3 + channel] * amount
                + background_pixels[index * 3 + channel] * (255 - amount)
                + 127
            ) // 255
            assert result[index * 3 + channel] == expected

    for index in (0, 3):
        assert result[index * 3 : index * 3 + 3] == source_pixels[index * 3 : index * 3 + 3]
    assert report == {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "background_sha256": hashlib.sha256(background).hexdigest(),
        "alpha_sha256": hashlib.sha256(alpha).hexdigest(),
        "protected_sha256": hashlib.sha256(protected).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "protected_count": 2,
        "changed_protected_pixels": 0,
        "max_abs_protected_change": 0,
        "fully_background_count": 1,
        "blended_count": 2,
    }
    assert (source, background, alpha, protected) == original_inputs
    assert compose_protected_product(*original_inputs)[0] == output


@pytest.mark.parametrize(
    ("role", "replacement"),
    [
        ("source", lambda: png("RGBA", (3, 2), bytes(range(24)))),
        ("background", lambda: png("RGB", (2, 2), bytes(range(12)))),
        ("alpha", lambda: png("L", (3, 2), bytes((0, 64, 128, 192, 224, 255)))),
        ("protected", lambda: png("L", (3, 2), bytes((255, 0, 0, 128, 0, 0)))),
    ],
)
def test_rejects_wrong_mode_dimensions_and_nonbinary_core(role: str, replacement) -> None:
    values = list(valid_inputs())
    values[("source", "background", "alpha", "protected").index(role)] = replacement()
    with pytest.raises(ValueError):
        compose_protected_product(*values)


@pytest.mark.parametrize(
    "mask_pixels",
    [
        bytes((0, 0, 0, 0, 0, 0)),
        bytes((255, 255, 255, 255, 255, 255)),
    ],
)
def test_rejects_empty_or_full_protected_mask(mask_pixels: bytes) -> None:
    values = list(valid_inputs())
    values[3] = png("L", (3, 2), mask_pixels)
    with pytest.raises(ValueError, match="protected mask"):
        compose_protected_product(*values)


def test_rejects_soft_alpha_inside_protected_core() -> None:
    values = list(valid_inputs())
    values[2] = png("L", (3, 2), bytes((254, 0, 128, 255, 64, 255)))
    with pytest.raises(ValueError, match="alpha"):
        compose_protected_product(*values)


def test_rejects_oversized_dimensions_bytes_and_corrupt_pngs() -> None:
    with pytest.raises(ValueError, match="dimension"):
        compose_protected_product(
            png("RGB", (2049, 1), bytes(2049 * 3)),
            png("RGB", (2049, 1), bytes(2049 * 3)),
            png("L", (2049, 1), bytes(2049)),
            png("L", (2049, 1), bytes([255]) * 2048 + b"\0"),
        )
    values = list(valid_inputs())
    values[0] = b"\0" * (16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="byte limit"):
        compose_protected_product(*values)
    values = list(valid_inputs())
    values[0] = b"corrupt png"
    with pytest.raises(ValueError):
        compose_protected_product(*values)


def test_rejects_header_exceeding_pixel_limit_before_decoding_pixels() -> None:
    values = list(valid_inputs())
    values = [png_with_dimensions(data, 2048, 2048) for data in values]
    with pytest.raises(ValueError, match="4 million pixel"):
        compose_protected_product(*values)


def test_rejects_non_png_rgb_input() -> None:
    values = list(valid_inputs())
    values[0] = png("RGB", (3, 2), bytes(range(18)), image_format="JPEG")
    with pytest.raises(ValueError, match="PNG"):
        compose_protected_product(*values)
